import os
import json
import argparse
import logging
import pdb

import yaml
import random
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
import sys
sys.path.append("/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen")
# from autoregressive.models.gpt import GPT_XXL_speech, GPT_Small_speech, GPT_XL, MultiTaskImageSpeech
# from autoregressive.models.gpt_cosy import GPT_XXL_speech, GPT_Small_speech, GPT_XL, MultiTaskImageSpeech
from autoregressive.models.gpt_cosy_prompt import GPT_XXL_speech, GPT_Small_speech, GPT_XL, MultiTaskImageSpeech
from transformers import CLIPProcessor, CLIPModel


from transformers.optimization import get_linear_schedule_with_warmup
from collections import deque
from torch.optim import AdamW
from transformers import get_constant_schedule
from transformers import RobertaTokenizerFast
from utils_text import load_config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # Added for TensorBoard integration
import numpy as np
from dataset.t2s import DatasetT2S as Dataset
import shutil
from language.t5 import T5Embedder

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Define color codes for logging (optional)
RED = '\033[91m'
RESET = '\033[0m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'


# Function to set up logging to both console and file
def setup_logging(log_file_path, rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # To prevent multiple handlers in DDP
    if not logger.handlers:
        # Formatter for the logs
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Console handler (only for rank 0)
        if rank == 0:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # File handler (all ranks can log, or only rank 0 based on preference)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fine-tune mBART50 for multilingual translation to English units."
    )
    parser.add_argument('--configs_training', type=str, required=True, help='Path to the YAML config file.')
    parser.add_argument('--config_model', type=str, required=True, help='Path folder to the JSON config')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with smaller datasets.')
    # No need for DDP-specific arguments when using torchrun
    args = parser.parse_args()
    return args


def save_model(model, tokenizer, path2save, logger, args, config):
    os.makedirs(path2save, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
    }, os.path.join(path2save, 'model.pth'))

def process_data_forward(batch, device=None, task="TTS",):
    input_ids, attention_mask, labels = None, None, None
    if task == "TTS":
        input_ids = batch['input_ids_tts']
        attention_mask = batch['attention_mask_tts']
        labels = batch['labels_tts']
    if task == "ASR":
        input_ids = batch['input_ids_asr']
        attention_mask = batch['attention_mask_asr']
        labels = batch['labels_asr']
    # convert to tensors
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return input_ids, attention_mask, labels


def main():
    args = parse_arguments()

    # Initialize Distributed Process Group
    # torchrun sets the following environment variables:
    # RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if world_size > 1:
        # Initialize the process group
        dist.init_process_group(backend='nccl', init_method='env://')

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    config = load_config(args.configs_training)
    # Create output directory if it doesn't exist (only rank 0)
    path2save = config['training']['output_dir']
    if rank == 0:
        os.makedirs(path2save, exist_ok=True)

    # Set up logging to both console and file (only rank 0 logs to console)
    log_file = os.path.join(path2save, 'training.log')
    logger = setup_logging(log_file_path=log_file, rank=rank)
    if rank == 0:
        logger.info("Logging is set up. Logs will be saved to both console and file.")

    # Initialize TensorBoard SummaryWriter (only rank 0)
    if rank == 0:
        tb_log_dir = os.path.join(path2save, 'tensorboard_logs')
        writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logging is set up. Logs will be saved to {tb_log_dir}")
    else:
        writer = None  # Other ranks do not log to TensorBoard

    # Save configuration to output directory (only rank 0)
    if rank == 0:
        with open(os.path.join(path2save, "configs_training.yaml"), 'w') as f:
            yaml.dump(config, f)

    # Initialize tokenizer and model
    # model_name = config['model']['name']
    # tokenizer_name = config['tokenizer']['name']
    ###############################
    tokenizer_path = "/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/pretrained_models/t5-ckpt-v2/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    ### load image generation model
    latent_size = config['model_config']['image_size'] // config['model_config']['downsample_size']
    img_model = GPT_XL(
        block_size=latent_size ** 2,
        vocab_size=config['image_config']['vocab_size'],
        cls_token_num=config['image_config']['cls_token_num'],
        model_type=config['image_config']['gpt_type'],
    ).to(device)
    # Load the model weights
    img_model_path = "/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/t2i_XL_stage2_512.pt"
    checkpoint = torch.load(img_model_path, map_location="cpu")
    img_model.load_state_dict(checkpoint['model'], strict=True)
    print(f"{RED}Loaded image generation model from: {img_model_path} {RESET}")

    model = MultiTaskImageSpeech(
        pretrained_image_model=img_model,
        text_vocab_size=config['speech_config']['text_vocab_size'],
        speech_vocab_size=config['speech_config']['vocab_speech_size'],
        n_speech_extra_layers=config['speech_config']['n_speech_extra_layers'],
        image_backbone_tuning_mode=config['model_config']['image_backbone_tuning_mode'],
        lora_alpha=config['model_config']['lora_alpha'],
        lora_rank=config['model_config']['lora_rank'],
    )
    model.to(device)

    #### Load pretrained model to fine-tune EARS
    pretrained_checkpoint = "/home/ldap-users/quangchung-t/Code/new_explore_multimodel/LlamaGen/result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce/LibriTTS_1e4_4Layer_16alpha_16rank_BS14_RemoveDup_KeepPunctuation/model_avg.pth"
    checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
    if 'module.' in list(checkpoint['model'].keys())[0]:
        new_state_dict = {}
        for k, v in checkpoint['model'].items():
            new_state_dict[k.replace('module.', '')] = v
        checkpoint['model'] = new_state_dict
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"{GREEN}Model loaded from {pretrained_checkpoint}{RESET}")
    #### Load pretrained model to fine-tune EARS

    # CLIP prompt embeddings store (expects per-tag vectors of dim 512)
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    # fronzen CLIP model
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval().to(device)

    logger.info(model)
    # Wrap the model with DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Load and preprocess JSON dataset
    train_dataset = Dataset(tokenizer, "train", config, args)
    dev_dataset = Dataset(tokenizer, "dev", config, args)

    if rank == 0:
        logger.info("Created PyTorch datasets.")

    # Create Distributed Samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None
        dev_sampler = None

    # Create DataLoaders
    # logger.info(f"{RED}Total parameters: {pytorch_total_params}{RESET}")
    # logger.info(f"{RED}Trainable parameters: {pytorch_total_params_trainable}{RESET}")
    logger.info(F"{RED}Warning: num_workers is set to {config['training']['num_workers']}{RESET}")
    logger.info(F"{RED}Warning: learning_rate is set to {config['training']['learning_rate']}{RESET}")
    train_loader = DataLoader(train_dataset, batch_size=int(config['training']['per_device_train_batch_size']),
                              sampler=train_sampler, shuffle=(train_sampler is None), pin_memory=True, collate_fn=train_dataset.collate_fn,
                              num_workers=int(config['training']['num_workers']))
    dev_loader = DataLoader(dev_dataset, batch_size=int(config['training']['per_device_eval_batch_size']),
                            sampler=dev_sampler, shuffle=False, pin_memory=True, collate_fn=dev_dataset.collate_fn,
                            num_workers=int(config['training']['num_workers']))
    if rank == 0:
        logger.info("Created DataLoaders for training, validation, and test sets.")


    # Define optimizer and scheduler
    trainable_params = []
    trainable_names = []
    for name, param in model.named_parameters():
        number_of_params = param.numel()
        if param.requires_grad:
            trainable_params.append(param)
            trainable_names.append(name)
            if rank == 0:
                logger.info(f"{RED}Trainable parameter: {name} - {param.shape} - {number_of_params} params{RESET}")
        else:
            if rank == 0:
                logger.info(f"{GREEN}Non-trainable parameter: {name} - {param.shape} - {number_of_params} params{RESET}")

    if rank == 0:
        logger.info(f"Model and tokenizer initialized. Device: {device}")
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{RED}{'Total parameters:'.ljust(20)}     {pytorch_total_params}{RESET}")
        logger.info(f"{RED}{'Trainable parameters:'.ljust(20)} {pytorch_total_params_trainable}{RESET}")

    optimizer = AdamW(trainable_params, lr=float(config['training']['learning_rate']),
                      betas=(float(config['training'].get('adam_beta1', 0.9)),
                             float(config['training'].get('adam_beta2', 0.999))),
                      eps=float(config['training'].get('adam_epsilon', 1e-8)),
                      weight_decay=float(config['training']['weight_decay']))

    num_epochs = int(config['training']['num_train_epochs'])
    total_steps = len(train_loader) * num_epochs

    ###############################
    scheduler = get_constant_schedule(optimizer)

    # Early Stopping parameters
    early_stopping_patience = int(config['training'].get('early_stopping_patience', 3))
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Track iterations and checkpoints
    global_step = 0
    checkpoint_queue = deque()

    primary_loss_total = 0.0
    primary_loss_tts_total = 0.0
    primary_loss_asr_total = 0.0
    if args.debug:
        save_iteration = config['debug']['save_iterations']
    else:
        save_iteration = config['training']['save_iterations']

    training_tts = config['task_training']['TTS']
    training_asr = config['task_training']['ASR']
    # if both is false, raise
    if not training_tts and not training_asr:
        raise ValueError("At least one of 'TTS' or 'ASR' must be enabled for training.")
    logger.info(f"{GREEN}Training TTS: {training_tts}{RESET}")
    logger.info(f"{GREEN}Training ASR: {training_asr}{RESET}")
    logger.info(f"{BLUE}Save to {config['training']['output_dir']}{RESET}")
    logger.info(f"{YELLOW}Single Training{RESET}")

    while global_step < total_steps:
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(global_step // len(train_sampler))  # Shuffle data periodically

        model.train()
        progress_bar = tqdm(train_loader, desc="Training", leave=False, disable=(rank != 0))
        for batch in progress_bar:
            primary_loss_tts = torch.tensor(0.0, device=device)
            primary_loss_asr = torch.tensor(0.0, device=device)
            prompt_instructions = batch['prompt_instructions']

            # extract style embedding from CLIP text encoder
            promt_inputs = clip_processor(text=prompt_instructions, return_tensors="pt", padding=True,
                                               truncation=True).to(device=device)
            with torch.no_grad():
                style_embeddings = clip_model.get_text_features(**promt_inputs)  # [B, 512]

            if training_tts:
                input_ids, attention_mask, labels = process_data_forward(batch, device, task="TTS")
                outputs = model.speech_forward(idx=input_ids, mask=attention_mask, targets=labels, style_embeddings=style_embeddings)
                primary_loss_tts = outputs.get("loss", torch.tensor(0.0, device=device))
                if primary_loss_tts.dim() > 0:
                    primary_loss_tts = primary_loss_tts.mean()

            if training_asr:
                input_ids, attention_mask, labels = process_data_forward(batch, device, task="ASR")
                outputs = model.speech_forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, task="ASR")
                primary_loss_asr = outputs.get("loss", torch.tensor(0.0, device=device))
                if primary_loss_asr.dim() > 0:
                    primary_loss_asr = primary_loss_asr.mean()
            # Combine losses
            primary_loss = (primary_loss_tts + primary_loss_asr)

            primary_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            primary_loss_total += primary_loss.item()
            primary_loss_tts_total += primary_loss_tts.item()
            primary_loss_asr_total += primary_loss_asr.item()

            global_step += 1
            progress_bar.set_description(f"Training {global_step}: | Loss-TTS: {primary_loss_tts.item():.4f} | Loss-ASR: {primary_loss_asr.item():.4f}")

            # Evaluate and save every X iterations
            if global_step % save_iteration == 0:
                if rank == 0:
                    avg_primary_loss = primary_loss_total / global_step
                    avg_primary_loss_tts = primary_loss_tts_total / global_step
                    avg_primary_loss_asr = primary_loss_asr_total / global_step

                    logger.info(f"Iteration {global_step}: Train Average Primary loss: {avg_primary_loss:.4f}")
                    logger.info(f"Iteration {global_step}: Train Average Primary loss TTS: {avg_primary_loss_tts:.4f}")
                    logger.info(f"Iteration {global_step}: Train Average Primary loss ASR: {avg_primary_loss_asr:.4f}")

                    if writer:
                        writer.add_scalar('Loss/Train Primary', avg_primary_loss, global_step)
                        writer.add_scalar('Loss/Train Primary TTS', avg_primary_loss_tts, global_step)
                        writer.add_scalar('Loss/Train Primary ASR', avg_primary_loss_asr, global_step)

                # Validation
                model.eval()
                primary_loss_tts_total_val = 0.0
                primary_loss_asr_total_val = 0.0

                with torch.no_grad():
                    for batch in tqdm(dev_loader, desc="Validation", leave=False, disable=(rank != 0)):
                        primary_loss_tts = torch.tensor(0.0, device=device)
                        primary_loss_asr = torch.tensor(0.0, device=device)
                        prompt_instructions = batch['prompt_instructions']
                        # extract style embedding from CLIP text encoder
                        # print(prompt_instructions)
                        promt_inputs = clip_processor(text=prompt_instructions, return_tensors="pt", padding=True,
                                                      truncation=True).to(device=device)
                        with torch.no_grad():
                            style_embeddings = clip_model.get_text_features(**promt_inputs)  # [B, 512]

                        if training_tts:
                            input_ids, attention_mask, labels = process_data_forward(batch, device, task="TTS")
                            outputs = model.speech_forward(idx=input_ids, mask=attention_mask, targets=labels, style_embeddings=style_embeddings)
                            primary_loss_tts = outputs.get("loss", torch.tensor(0.0, device=device))
                            if primary_loss_tts.dim() > 0:
                                primary_loss_tts = primary_loss_tts.mean()

                        if training_asr:
                            input_ids, attention_mask, labels = process_data_forward(batch, device, task="ASR")
                            outputs = model.speech_forward(input_ids=input_ids, attention_mask=attention_mask,
                                            labels=labels, task="ASR")
                            primary_loss_asr = outputs.get("loss", torch.tensor(0.0, device=device))
                            if primary_loss_asr.dim() > 0:
                                primary_loss_asr = primary_loss_asr.mean()

                        primary_loss_tts_total_val += primary_loss_tts.item()
                        primary_loss_asr_total_val += primary_loss_asr.item()

                if rank == 0:
                    avg_primary_loss_tts = primary_loss_tts_total_val / len(dev_loader)
                    avg_primary_loss_asr = primary_loss_asr_total_val / len(dev_loader)

                    logger.info(
                        f"Iteration {global_step}: Validation Average Primary loss TTS: {avg_primary_loss_tts:.4f}")
                    logger.info(
                        f"Iteration {global_step}: Validation Average Primary loss ASR: {avg_primary_loss_asr:.4f}")
                    if writer:
                        writer.add_scalar('Loss/Validation Primary TTS', avg_primary_loss_tts, global_step)
                        writer.add_scalar('Loss/Validation Primary ASR', avg_primary_loss_asr, global_step)

                    # # Early stopping and checkpointing
                    if config["loss_target"] == "TTS":
                        avg_primary_loss_dev = avg_primary_loss_tts
                    elif config["loss_target"] == "ASR":
                        avg_primary_loss_dev = avg_primary_loss_asr
                    else:
                        avg_primary_loss_dev = avg_primary_loss_tts + avg_primary_loss_asr
                    if avg_primary_loss_dev < best_val_loss:
                        best_val_loss = avg_primary_loss_dev
                        epochs_no_improve = 0
                        logger.info(f"{GREEN}Best model updated at iteration {global_step}{RESET}")

                        # Save checkpoint
                        checkpoint_path = os.path.join(path2save, f"checkpoint_iter_{global_step}")
                        save_model(model, tokenizer, checkpoint_path, logger, args, config)
                        checkpoint_queue.append(checkpoint_path)
                        logger.info(f"{GREEN}Checkpoint saved at iteration {global_step}{RESET}")
                        if len(checkpoint_queue) > config['training'].get('keep_last_n_checkpoints', 3):
                            oldest_checkpoint = checkpoint_queue.popleft()
                            # remove folder
                            if os.path.isdir(oldest_checkpoint):
                                shutil.rmtree(oldest_checkpoint)
                                logger.info(f"{CYAN}Removed oldest checkpoint: {oldest_checkpoint}{RESET}")
                    else:
                        epochs_no_improve += 1
                        logger.info(
                            f"{YELLOW}No improvement in validation loss for {epochs_no_improve} iteration(s).{RESET}")
                        if epochs_no_improve >= early_stopping_patience:
                            logger.info(f"{RED}Early stopping triggered at iteration {global_step}.{RESET}")
                            logger.info(f"{RED}Training completed.{RESET}")

                # Reset training losses for the next interval
                model.train()

                # Synchronize all processes to ensure they proceed together
                if world_size > 1:
                    dist.barrier()

    if rank == 0 and writer:
        writer.close()
        logger.info("TensorBoard writer closed.")
        logger.info(f"{RED}Early stopping triggered at iteration {global_step}.{RESET}")
        logger.info(f"{RED}Training completed.{RESET}")
        exit()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()