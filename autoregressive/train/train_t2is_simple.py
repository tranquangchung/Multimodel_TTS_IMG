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

    # --- DDP setup ---
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # --- Config, I/O, logging, TB ---
    config = load_config(args.configs_training)
    path2save = config['training']['output_dir']
    if rank == 0:
        os.makedirs(path2save, exist_ok=True)
    log_file = os.path.join(path2save, 'training.log')
    logger = setup_logging(log_file_path=log_file, rank=rank)
    if rank == 0:
        if args.debug:
            logger.info(f"{YELLOW}DEBUG mode enabled{RESET}")
        logger.info("Logging is set up. Logs will be saved to both console and file.")
        tb_log_dir = os.path.join(path2save, 'tensorboard_logs')
        writer = SummaryWriter(log_dir=tb_log_dir)
        with open(os.path.join(path2save, "configs_training.yaml"), 'w') as f:
            yaml.dump(config, f)
    else:
        writer = None

    # --- Tokenizer & models (giữ nguyên như cũ) ---
    tokenizer_path = "/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/pretrained_models/t5-ckpt-v2/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    latent_size = config['model_config']['image_size'] // config['model_config']['downsample_size']
    img_model = GPT_XL(
        block_size=latent_size ** 2,
        vocab_size=config['image_config']['vocab_size'],
        cls_token_num=config['image_config']['cls_token_num'],
        model_type=config['image_config']['gpt_type'],
    ).to(device)
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
    ).to(device)

    pretrained_checkpoint = "/home/ldap-users/quangchung-t/Code/new_explore_multimodel/LlamaGen/result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce/LibriTTS_1e4_4Layer_16alpha_16rank_BS14_NoRemoveDup/model_avg.pth"
    checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
    if 'module.' in list(checkpoint['model'].keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        checkpoint['model'] = new_state_dict
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"{GREEN}Model loaded from {pretrained_checkpoint}{RESET}")

    logger.info(model)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # --- Data ---
    train_dataset = Dataset(tokenizer, "train", config, args)
    dev_dataset = Dataset(tokenizer, "dev", config, args)
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dev_sampler = DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        dev_sampler = None

    logger.info(F"{RED}Warning: num_workers is set to {config['training']['num_workers']}{RESET}")
    logger.info(F"{RED}Warning: learning_rate is set to {config['training']['learning_rate']}{RESET}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['training']['per_device_train_batch_size']),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=int(config['training']['num_workers'])
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=int(config['training']['per_device_eval_batch_size']),
        sampler=dev_sampler,
        shuffle=False,
        pin_memory=True,
        collate_fn=dev_dataset.collate_fn,
        num_workers=int(config['training']['num_workers'])
    )
    if rank == 0:
        logger.info("Created DataLoaders.")

    # --- Optimizer & schedule ---
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

    optimizer = AdamW(
        trainable_params,
        lr=float(config['training']['learning_rate']),
        betas=(float(config['training'].get('adam_beta1', 0.9)),
               float(config['training'].get('adam_beta2', 0.999))),
        eps=float(config['training'].get('adam_epsilon', 1e-8)),
        weight_decay=float(config['training']['weight_decay'])
    )
    scheduler = get_constant_schedule(optimizer)

    num_epochs = int(config['training']['num_train_epochs'])
    total_steps = len(train_loader) * num_epochs
    save_iteration = config['debug']['save_iterations'] if args.debug else config['training']['save_iterations']

    best_val_loss = float('inf')
    early_stopping_patience = int(config['training'].get('early_stopping_patience', 3))
    epochs_no_improve = 0
    global_step = 0
    checkpoint_queue = deque()

    # --- Standard training loop: single forward, single loss ---
    while global_step < total_steps:
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(global_step // max(1, len(train_loader)))  # shuffle per epoch-ish

        model.train()
        progress_bar = tqdm(train_loader, desc="Training", leave=False, disable=(rank != 0))
        for batch in progress_bar:
            input_ids = batch.get('input_ids_tts').to(device, non_blocking=True)
            attention_mask = batch.get('attention_mask_tts').to(device, non_blocking=True)
            labels = batch.get('labels_tts').to(device, non_blocking=True)
            prompt_instructions = batch.get('prompt_instructions', None)

            optimizer.zero_grad(set_to_none=True)
            outputs = model.speech_forward(
                idx=input_ids,
                mask=attention_mask,
                targets=labels,
                prompt_instructions=prompt_instructions
            )
            loss = outputs['loss']
            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if rank == 0:
                progress_bar.set_description(f"Training {global_step} | Loss: {loss.item():.4f}")
                if writer:
                    writer.add_scalar('Loss/Train', float(loss.item()), global_step)

            # --- Eval + checkpoint theo chu kỳ ---
            if global_step % save_iteration == 0:
                model.eval()
                val_loss_sum, val_count = 0.0, 0
                with torch.no_grad():
                    for vbatch in tqdm(dev_loader, desc="Validation", leave=False, disable=(rank != 0)):
                        vid = vbatch.get('input_ids_tts').to(device, non_blocking=True)
                        vmsk = vbatch.get('attention_mask_tts').to(device, non_blocking=True)
                        vlbl = vbatch.get('labels_tts').to(device, non_blocking=True)
                        vpi = vbatch.get('prompt_instructions', None)
                        vout = model.speech_forward(idx=vid, mask=vmsk, targets=vlbl, prompt_instructions=vpi)
                        vloss = vout['loss']
                        if vloss.dim() > 0:
                            vloss = vloss.mean()
                        val_loss_sum += float(vloss.item())
                        val_count += 1

                if rank == 0 and val_count > 0:
                    avg_val = val_loss_sum / val_count
                    logger.info(f"Iteration {global_step}: Validation Loss: {avg_val:.4f}")
                    if writer:
                        writer.add_scalar('Loss/Val', avg_val, global_step)

                    if avg_val < best_val_loss:
                        best_val_loss = avg_val
                        epochs_no_improve = 0
                        ckpt_dir = os.path.join(path2save, f"checkpoint_iter_{global_step}")
                        save_model(model, tokenizer, ckpt_dir, logger, args, config)
                        checkpoint_queue.append(ckpt_dir)
                        logger.info(f"{GREEN}Checkpoint saved at iteration {global_step}{RESET}")
                        # giữ N checkpoint gần nhất
                        keep_n = int(config['training'].get('keep_last_n_checkpoints', 3))
                        while len(checkpoint_queue) > keep_n:
                            old = checkpoint_queue.popleft()
                            if os.path.isdir(old):
                                shutil.rmtree(old)
                                logger.info(f"{CYAN}Removed oldest checkpoint: {old}{RESET}")
                    else:
                        epochs_no_improve += 1
                        logger.info(f"{YELLOW}No improvement for {epochs_no_improve} iteration(s).{RESET}")
                        if epochs_no_improve >= early_stopping_patience:
                            logger.info(f"{RED}Early stopping at iteration {global_step}.{RESET}")
                            break

                model.train()
                if world_size > 1:
                    dist.barrier()

        # nếu early stop trong epoch
        if epochs_no_improve >= early_stopping_patience:
            break

    if rank == 0 and writer:
        writer.close()
        logger.info(f"{RED}Training completed.{RESET}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()