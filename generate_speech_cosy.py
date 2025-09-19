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
sys.path.append('/home/ldap-users/s2220411/Code/new_explore_tts/CosyVoice')
sys.path.append('/home/ldap-users/s2220411/Code/new_explore_tts/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.model import CosyVoice2Model

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
from hyperpyyaml import load_hyperpyyaml


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
    distinct_numbers = [x for x in distinct_numbers if int(x) < 6561]
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

# text_sources = [
#     "A small rabbit hopped through the meadow after the rain, its damp fur shining. Under a leaf, it sat still, listening with wonder.",
#     "An owl perched high on a branch as stars glowed above. Its bright eyes watched the dark path, wings ready to fly.",
#     "A turtle moved slowly along the shore, waves touching its shell. It carried peace within, unhurried toward the sea.",
#     "A gentle deer stood at the forest’s edge, golden fields stretching wide. Thin but strong, it breathed in the fading light.",
#     "A young bear sat by a stream, paws dipping into cool water. With bright eyes, it played as the forest hummed around it.",
#     "A cat stretched on the windowsill as rain tapped softly on the glass. Yawning, it curled into a ball, warm and safe.",
#     "A dog raced across the bright field, ears flying back. Stopping to sniff, it dashed forward again, chasing joy.",
#     "A tiny bird puffed its feathers against the breeze as the morning sky glowed. Tilting its head, it sang a clear note of hope.",
#     "A squirrel scurried along a branch, clutching an acorn tight. Without fear, it leapt to the next branch with ease.",
#     "A little hedgehog shuffled through dusk grass, leaving soft trails in the earth. When the breeze rustled, it curled slightly before moving on."
# ]

text_sources = [
    # "The little rabbit found a shiny red apple in the grass. He happily shared it with his best friend, the squirrel."
    # "A hungry fox was walking past a vineyard when he saw some ripe, juicy grapes hanging from a vine. They looked so delicious! The fox jumped high to grab them but couldn’t reach them, no matter how hard he tried. Tired and frustrated, the fox walked away, grumbling, “Those grapes are probably sour anyway!” But deep down, he knew he had just given up too quickly."
    "A hare and a tortoise were friends. The hare bragged he was the fastest and challenged the tortoise to a race. The hare ran quickly and stopped to nap. The tortoise kept going slowly and steadily. When the hare woke up, the tortoise had already crossed the finish line and won."
]

def generate_speech(
    model,
    tokenizer,
    text_source,
    device,
    configs,
    frontend,
    cosyvoice_model,
    lang=None,
    path2save_audio=None,
):
    """
    AR model sinh unit -> CosyVoice2.flow (mel) -> CosyVoice2.hift (audio).
    Lưu WAV ở path2save_audio (24 kHz).
    """
    assert lang is not None, "lang phải được set, ví dụ 'English'"
    assert frontend is not None and cosyvoice_model is not None, "frontend/cosyvoice_model chưa được khởi tạo"

    # --- cấu hình chung
    resample_rate = 24000
    fp16 = bool(torch.cuda.is_available())

    # --- BOS/EOS cho TTS
    tts_turn = configs['custom_data']['bos_tts'][lang]
    tts_eos  = configs['custom_data']['eos_tts']
    print(f"{YELLOW}tts_turn: {tts_turn}, tts_eos: {tts_eos}{RESET}")

    encoded_inputs = tokenizer(text_source, return_tensors='pt', padding=True, truncation=True).to(device)
    append_turn = torch.tensor([[tts_turn]], device=encoded_inputs['input_ids'].device)
    append_mask = torch.tensor([[1]], device=encoded_inputs['attention_mask'].device)
    encoded_inputs['input_ids']     = torch.cat([encoded_inputs['input_ids'], append_turn], dim=1)
    encoded_inputs['attention_mask'] = torch.cat([encoded_inputs['attention_mask'], append_mask], dim=1)
    encoded_inputs_length = encoded_inputs['input_ids'].shape[1]

    # --- sinh unit bằng AR head
    max_new_tokens = configs.get('inference', {}).get('max_length_unit', 512)
    with torch.no_grad():
        outs = model.speech_generate(
            encoded_inputs['input_ids'],
            max_new_tokens=max_new_tokens,
        )
    outs = outs.tolist()[0][encoded_inputs_length:]  # cắt phần prompt

    # --- khử lặp + giới hạn vocab
    predict_unit = remove_consecutive_duplicates(outs)
    # predict_unit = outs
    if len(predict_unit) == 0:
        print(f"{RED}predict_unit rỗng -> bỏ qua{RESET}")
        return

    # --- chuẩn bị prompt cho CosyVoice2
    prompt_wav_path = '/home/ldap-users/Share/Data/librispeech/test-clean/1089/134686/1089-134686-0035.flac'

    prompt_wav_16k = load_wav(prompt_wav_path, 16000)  # [T] hoặc numpy
    if not torch.is_tensor(prompt_wav_16k):
        prompt_wav_16k = torch.from_numpy(prompt_wav_16k)
    if prompt_wav_16k.dim() == 1:
        prompt_wav_16k = prompt_wav_16k.unsqueeze(0)  # [1, T]

    prompt_speech_resample = ta.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_wav_16k)

    # Lấy đặc trưng/token/embedding từ frontend (CPU), rồi đưa sang device khi cần
    speech_feat, speech_feat_len         = frontend._extract_speech_feat(prompt_speech_resample)
    prompt_speech_token, speech_token_len = frontend._extract_speech_token(prompt_wav_16k)
    # Đồng bộ chiều dài theo logic CosyVoice
    token_len = min(int(speech_feat.shape[1] // 2), prompt_speech_token.shape[1])
    speech_feat, speech_feat_len[:]           = speech_feat[:, :2 * token_len], 2 * token_len
    prompt_speech_token, speech_token_len[:]  = prompt_speech_token[:, :token_len], token_len
    embedding = frontend._extract_spk_embedding(prompt_wav_16k)  # [1, spk_dim]

    # --- Flow (mel) + HIFT (audio)
    speech_token_predict = torch.tensor(predict_unit, dtype=torch.int32, device=device).unsqueeze(0)  # [1, T]
    hift_cache_source = torch.zeros(1, 1, 0, device=device)
    with torch.cuda.amp.autocast(enabled=fp16):
        tts_mel, _ = cosyvoice_model.flow.inference(
            token=speech_token_predict,
            token_len=torch.tensor([speech_token_predict.shape[1]], dtype=torch.int32, device=device),
            prompt_token=prompt_speech_token.to(device),
            prompt_token_len=torch.tensor([prompt_speech_token.shape[1]], dtype=torch.int32, device=device),
            prompt_feat=speech_feat.to(device),
            prompt_feat_len=torch.tensor([speech_feat.shape[1]], dtype=torch.int32, device=device),
            embedding=embedding.to(device),
            streaming=False,
            finalize=True,
        )

    audio, _ = cosyvoice_model.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)

    # --- lưu WAV 24kHz
    if path2save_audio is not None:
        audio = audio.detach().cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [C, T]
        audio = audio.to(torch.float32).clamp_(-1, 1)
        ta.save(path2save_audio, audio, resample_rate)
        print(f"{GREEN}Saved: {path2save_audio}{RESET}")


def perform_inference(model, tokenizer, test_data, frontend, cosyvoice_model, configs, lang=None):
    path2save_root = os.path.join(configs['training']['output_dir'], args.folder2save)
    if not os.path.exists(path2save_root):
        os.makedirs(path2save_root)
    # Load ground truth
    groundtruth = []
    transcript_ars = []
    for index, item in enumerate(test_data):
        text_source = item['transcript'].strip()
        # text_source = text_sources[index % len(text_sources)]
        path2save_audio = os.path.join(path2save_root, f"audio_{index}.wav")
        generate_speech(
            model=model,
            tokenizer=tokenizer,
            text_source=text_source,
            device=device,
            configs=configs,
            frontend=frontend,
            cosyvoice_model=cosyvoice_model,
            lang=lang,
            path2save_audio=path2save_audio
        )
        #### transcription using ASR model
        transcript = transcription_whisper(path2save_audio, language=lang)
        print(f"{'text source'.ljust(20)}: {text_source}")
        print(f"{'transcript (ASR)'.ljust(20)}: {transcript}")
        groundtruth.append(remove_special_characters(text_source))
        transcript_ars.append(remove_special_characters(transcript))
        if args.debug and index >= 2:
            break
        if index % 100 == 0:
            print(f"Processed {index} samples")
        if index > 1000:
            break
    ############################
    print(f"{GREEN}Done. Check out {path2save_root}{RESET}")
    # evaluate
    score = evaluate_transcription(groundtruth, transcript_ars)
    # save score
    with open(os.path.join(path2save_root, f"{lang}_score.json"), 'w') as fp:
        json.dump(score, fp, indent=4)
    print(f"Score: {score}")

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
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"{GREEN}Model loaded from {pretrained_checkpoint}{RESET}")
    ############################
    lang = "English"  # Default language
    print(f"{YELLOW}Processing {lang}{RESET}")
    path_dir = "/home/ldap-users/s2220411/Code/new_explore_tts/MachineSpeechChain_ASRU25/dataset/LibriTTS"
    path_file = os.path.join(path_dir, lang, f"{args.dataset}.json")
    print(f"Loading test data from {path_file}")
    with open(path_file, 'r') as f:
        test_data = json.load(f)
        print(f"lang: {len(test_data)} samples")

    ## load cosyvoice2
    model_dir = 'pretrained_models/CosyVoice2-0.5B'
    hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f,
                                   overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                 configs['feat_extractor'],
                                 '{}/campplus.onnx'.format(model_dir),
                                 '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                 '{}/spk2info.pt'.format(model_dir),
                                 configs['allowed_special'])
    cosyvoice_model = CosyVoice2Model(
        configs['llm'], configs['flow'], configs['hift'], False
    )
    cosyvoice_model.load('{}/llm.pt'.format(model_dir),
                         '{}/flow.pt'.format(model_dir),
                         '{}/hift.pt'.format(model_dir))
    print("load frontend done")

    perform_inference(model, tokenizer, test_data, frontend, cosyvoice_model, configs=config, lang=lang)


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
