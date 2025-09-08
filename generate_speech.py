#!/usr/bin/env python
# coding: utf-8
# /home/ldap-users/s2220411/miniconda3/envs/tts_lowresource/lib/python3.10/site-packages/transformers/generation/utils.py
import os
import json
import argparse
import pdb

import yaml
import torch
from transformers import AutoTokenizer, GenerationConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from scipy.io import wavfile  # Used to read audio files
import librosa

### this is the code from the Grad-TTS
import sys

# from autoregressive.models.gpt import GPT_XXL_speech, MultiTaskImageSpeech, GPT_XL
from autoregressive.models.gpt_cosy import GPT_XXL_speech, MultiTaskImageSpeech, GPT_XL
sys.path.append('/home/ldap-users/s2220411/Code/new_explore_tts/Speech-Backbones/Grad-TTS')
sys.path.append('/home/ldap-users/s2220411/Code/new_explore_tts/Speech-Backbones/Grad-TTS/hifi-gan')
from model import GradTTS
import params
from env import AttrDict
from models import Generator as HiFiGAN
from scipy.io.wavfile import write
from transformers import PretrainedConfig
import sacrebleu
import whisper
import jiwer
import numtext as nt
import re
import torchaudio as ta
import time
import random
import glob

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


# ANSI color codes for terminal output
RED = '\033[91m'
RESET = '\033[0m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
_whitespace_re = re.compile(r'\s+')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

### this is the code from the Grad-TTS
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_whisper = whisper.load_model("large-v2")

language_mapping_whisper = {
    "Chinese": "zh",
    "Dutch": "nl",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hungarian": "hu",
    "Japanese": "ja",
    "Russian": "ru",
    "Spanish": "es",
    "English": "en",
}

def transcription_whisper(audio_file_path, language="English"):
    lang_whisper = language_mapping_whisper[language]
    result = model_whisper.transcribe(audio_file_path, language=lang_whisper)
    transcription = result["text"]
    return transcription.strip()

def evaluate_transcription(groundtruth, transcript_ars):
    wer = jiwer.wer(groundtruth, transcript_ars)
    cer = jiwer.cer(groundtruth, transcript_ars)
    return {
        "wer": wer,
        "cer": cer,
    }


def remove_consecutive_duplicates(number_sequence):
    distinct_numbers = [number_sequence[i] for i in range(len(number_sequence)) if i == 0 or number_sequence[i] != number_sequence[i - 1]]
    # remove token overrange 0->999
    distinct_numbers = [x for x in distinct_numbers if int(x) < 1000]
    return distinct_numbers

def get_mel(filepath):
    y, sr = librosa.load(filepath)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

def compare_L2(filepath1, filepath2):
    mel1 = get_mel(filepath1)
    mel2 = get_mel(filepath2)
    _, T1 = mel1.shape
    _, T2 = mel2.shape
    T = min(T1, T2)
    mel1 = mel1[:, :T]
    mel2 = mel2[:, :T]
    l2_distance = np.linalg.norm(mel1 - mel2, ord=2)
    return l2_distance

def load_config(config_path):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_groundtruth(path_groundtruth):
    # Load ground truth
    with open(path_groundtruth, 'r') as f:
        lines = f.readlines()
        groundtruth = {}
        for line in lines:
            parts = line.strip().split('\t')
            groundtruth[parts[0]] = parts[1].lower()

    return groundtruth

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                  "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                  "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                  "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                  "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "'", "ʻ", "ˆ", "-"]

def remove_special_characters(text):
    def convert_match(match):
        number = match.group(0)
        return nt.convert(int(number))  # Convert number to text

    text = re.sub(r'\b\d+\b', convert_match, text)
    for char in CHARS_TO_IGNORE:
        text = text.replace(char.strip(), "")

    text = re.sub(_whitespace_re, ' ', text)
    return text.lower().strip()

text_sources = [
    "in the middle of a mountain meadow, a bear sits quietly, planting tiny seeds in a circle around him. ",
    "he wears round glasses and a patchy gardener’s hat. ",
    "as the sun rises, the seeds bloom instantly into colorful flowers—each one with a face that smiles and hums softly. ",
    "butterflies hover near him, and the bear carefully writes in a little notebook made of leaves.",
]

def perform_inference(model, tokenizer, test_data, device, generator, vocoder, configs, lang=None):
    # spk = torch.tensor([0]).cuda()
    spk = None
    tts_turn = configs['custom_data']['bos_tts'][lang]
    tts_eos = configs['custom_data']['eos_tts']
    print(f"{YELLOW}tts_turn: {tts_turn}, tts_eos: {tts_eos}{RESET}")

    root_path2save = os.path.join(configs['training']['output_dir'], args.folder2save)
    os.makedirs(root_path2save, exist_ok=True)
    # path2save wav
    path2save_wav = os.path.join(configs['training']['output_dir'], args.folder2save, "wav")
    os.makedirs(path2save_wav, exist_ok=True)
    # Load ground truth
    groundtruth = []
    transcript_ars = []
    predictions = []
    # tokenizer.src_lang = lang_code
    start_time = time.time()
    for index, item in tqdm(enumerate(test_data)):
        # text_source = remove_special_characters(item['transcript'])
        # text_source = remove_special_characters(text_sources[index % len(text_sources)])
        text_source = item['transcript']
        audio_path = item['audio_path']
        file_name = os.path.basename(audio_path).split(".")[0]
        encoded_inputs = tokenizer(text_source, return_tensors='pt', padding=True, truncation=True).to(device)
        encoded_inputs['input_ids'] = torch.cat(
            [encoded_inputs['input_ids'], torch.tensor([[tts_turn]], device='cuda:0')], dim=1)
        encoded_inputs['attention_mask'] = torch.cat(
            [encoded_inputs['attention_mask'], torch.tensor([[1]], device='cuda:0')], dim=1)
        encoded_inputs_length = encoded_inputs['input_ids'].shape[1]
        max_length = encoded_inputs_length + configs["preprocessing"]["max_length_unit"]
        # custom_config = GenerationConfig(
        #     eos_token_id=tts_eos,
        #     pad_token_id=tts_eos,
        #     max_length=max_length,  # Custom max length
        #     num_beams=configs['inference']['num_beams'],  # Custom number of beams
        #     early_stopping=True, # Early stopping
        # )

        if encoded_inputs_length < 5: continue
        ########################################
        print("*" * 20)
        print(f"{BLUE} TTS: {config['training']['output_dir']} --> processing index: {index}/{len(test_data)}, lang: {lang}, dataset: {args.dataset}{RESET}")
        with torch.no_grad():
            outputs = model.speech_generate(
                encoded_inputs['input_ids'],
                max_new_tokens=300,
            )
            # outputs = model.generate(**encoded_inputs, generation_config=custom_config)
            outputs = outputs.tolist()
            outputs = outputs[0][encoded_inputs_length:]
            # print(outputs)
            print(f"{'text source (GT)'.ljust(20)}: {text_source}")
            predict_unit = remove_consecutive_duplicates(outputs)
            predict_unit_str = " ".join([str(x) for x in predict_unit])
            # groundtruth_unit = item['unit'].split()
            # groundtruth_unit = remove_consecutive_duplicates(groundtruth_unit)
            # groundtruth_unit = " ".join([str(x) for x in groundtruth_unit])
            # print 10 first tokens
            # print(f"{'predict unit'.ljust(20)}: {predict_unit_str}")
            # print(f"{'groundtruth unit'.ljust(20)}: {groundtruth_unit}")
            ########################################
            x = torch.IntTensor(predict_unit).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).to(device)

            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=configs["inference_gradtts"]["n_timesteps"], temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91)
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            path2save_wavfile = os.path.join(path2save_wav, f"{index}_" + file_name + ".wav")
            write(path2save_wavfile, 22050, audio)
            #### transcription using ASR model
            transcript = transcription_whisper(path2save_wavfile, language=lang)
            transcript = remove_special_characters(transcript)
            print(f"{'transcript (ASR)'.ljust(20)}: {transcript}")
            groundtruth.append(text_source)
            transcript_ars.append(transcript)
            predict = {
                "transcript": text_source,
                "transcript_wav2vec": text_source,
                "transcript_predict": transcript,
                "unit_predict": predict_unit_str,
                "unit": item['unit'],
                "audio_path": audio_path,
                "language": "en_XX",
                "task": "TTS",
            }
            predictions.append(predict)
            #######################################
            # os.remove(path2save_wavfile)
            if args.debug:
                if index > 10: break
    end_time = time.time()
    print(f"{RED}Inference completed in {end_time - start_time:.2f} seconds.{RESET}")

    ############################
    print(f"{GREEN}Done. Check out {path2save_wav} folder for samples.{RESET}")
    # evaluate
    score = evaluate_transcription(groundtruth, transcript_ars)
    # save score
    with open(os.path.join(path2save_wav, f"{lang}_{args.dataset}_score.json"), 'w') as fp:
        json.dump(score, fp, indent=4)
    print(f"Score: {score}")
    # save predictions
    path2save = os.path.join(root_path2save, f"{lang}_{args.dataset}_prediction.json")
    with open(path2save, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print(f"{GREEN}Predictions saved to {path2save}{RESET}")

def inference(config):
    # 3. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{CYAN}Using device: {device}{RESET}")

    # 4. Initialize tokenizer and model
    tokenizer_path = "/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/pretrained_models/t5-ckpt/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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

    model = MultiTaskImageSpeech(
        pretrained_image_model=img_model,
        text_vocab_size=config['speech_config']['text_vocab_size'],
        speech_vocab_size=config['speech_config']['vocab_speech_size'],
        n_speech_extra_layers=config['speech_config']['n_speech_extra_layers'],
        image_backbone_tuning_mode=config['model_config']['image_backbone_tuning_mode'],
        lora_alpha=config['model_config']['lora_alpha'],
        lora_rank=config['model_config']['lora_rank'],
    )

    print(model)
    model.eval()
    model.to(device)
    out_dir = config['training']['output_dir']
    pretrained_checkpoint = os.path.join(out_dir, 'model_avg.pth')
    if not os.path.exists(pretrained_checkpoint):
        model_names = glob.glob(f"{out_dir}/checkpoint_iter_*")
        model_names.sort(key=lambda x: int(x.split("_")[-1]))
        pretrained_checkpoint = os.path.join(model_names[-1], "model.pth")
    print("Loading model from:", pretrained_checkpoint)
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    # remove "module."
    if 'module.' in list(checkpoint['model'].keys())[0]:
        new_state_dict = {}
        for k, v in checkpoint['model'].items():
            new_state_dict[k.replace('module.', '')] = v
        checkpoint['model'] = new_state_dict
    model.load_state_dict(checkpoint['model'], strict=True)
    print(f"{GREEN}Model loaded from {pretrained_checkpoint}{RESET}")
    ############################

    print('Initializing HiFi-GAN...')
    hifi_gan_config = config["inference_gradtts"]["hifi_gan_config"]
    hifi_gan_checkpoint = config["inference_gradtts"]["hifi_gan_checkpoint"]
    with open(hifi_gan_config) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(hifi_gan_checkpoint, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    # 8. Load test data
    # path_file = "/home/ldap-users/s2220411/Code/new_explore_tts/MachineSpeechChain_ASRU25/dataset/LibriTTS/English/test-clean.json"
    path_dir = config['dataset']['path_dir']
    if len(path_dir) > 1:
        raise ValueError(f"Path directory should be a single path, got {path_dir}")
    else:
        path_dir = path_dir[0]
    for lang in ["English"]:
        print(f"{YELLOW}Processing {lang}{RESET}")
        path_file = os.path.join(path_dir, lang, f"{args.dataset}.json")
        print(f"Loading test data from {path_file}")
        with open(path_file, 'r') as f:
            test_data = json.load(f)
            print(f"lang: {len(test_data)} samples")
            # split test_data into 3 parts
            # part_size = len(test_data) // 3
            # test_data = test_data[:part_size]
            # test_data = test_data[part_size:2 * part_size]
            # test_data = test_data[2 * part_size:]

        n_spks = 1
        generator = GradTTS(1001, n_spks, params.spk_emb_dim,
                            params.n_enc_channels, params.filter_channels,
                            params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                            params.enc_kernel, params.enc_dropout, params.window_size,
                            params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
        checkpoint = f"/home/ldap-users/s2220411/Code/new_explore_tts/Speech-Backbones/Grad-TTS/logs/CSS10_1K/{lang}/grad_250.pt"
        if not os.path.exists(checkpoint):
            print(f"{RED}Checkpoint {checkpoint} does not exist. Skipping...{RESET}")
            continue
        generator.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc))
        _ = generator.cuda().eval()

        perform_inference(model, tokenizer, test_data, device, generator, vocoder, configs=config, lang=lang)


if __name__ == "__main__":
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    parser.add_argument('--dataset', type=str, required=False, help='Dataset to use for inference.', default="test-clean.json")
    parser.add_argument('--folder2save', type=str, required=False, help='Language code to translate from.', default="prediction")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with smaller datasets.')
    args = parser.parse_args()
    print(args)
    # 2. Load configuration
    config = load_config(args.config)
    inference(config)
