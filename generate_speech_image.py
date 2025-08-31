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

from autoregressive.models.gpt import GPT_XXL_speech, MultiTaskImageSpeech, GPT_XL
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
from tokenizer.tokenizer_image.vq_model import VQ_16
from transformers import T5EncoderModel, AutoTokenizer
from autoregressive.models.generate import generate as generate_img_fn
from torchvision.utils import save_image

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

# text_sources = [
#     "[Anime style]: A small rabbit hopped through the meadow after the rain. Its ears twitched at every sound, and its fur was still damp. It found a quiet spot under a leaf and watched drops fall to the ground. The world felt large and calm, and the rabbit simply listened, heart beating soft with wonder.",
#     "An owl perched on a tall branch in the silent night. The stars glowed above, and the wind whispered through the trees. The owl’s eyes shone bright, watching the dark path below. It spread its wings slowly, ready to fly. For the owl, the night was not lonely, but full of secrets.",
#     "A turtle moved slowly along the shore, each step steady and sure. Waves rolled in and touched its shell before sliding back. The air smelled of salt, and the sand was warm underfoot. The turtle did not hurry. It carried a quiet world within, finding peace in every small movement toward the sea.",
#     "A gentle deer stood at the edge of the forest, nose lifted to the breeze. The field stretched wide, golden in the fading light. Its legs were thin but strong, ready to leap if needed. Yet in that moment, the deer simply stood still, breathing in the quiet beauty of the open air.",
#     "A young bear sat by a stream, paws dipping into the cool water. Small fish flashed like silver under the surface. The bear leaned close, eyes bright with hunger and play. The forest hummed softly around it, full of life. For the bear, the world was both a playground and a home.",
#     "A cat stretched on the windowsill, tail curling like a ribbon. Outside, the rain tapped softly on the glass. The cat’s eyes followed every drop, slow and calm. It gave a long yawn, then curled into a ball, listening to the gentle rhythm. In that small space, the cat felt warm and safe.",
#     "A dog ran across the field with ears flying back and tongue hanging out. The grass bent under its paws, and the sky was wide and bright. It stopped to sniff the ground, then dashed forward again, chasing nothing but joy. For the dog, every open space was a new adventure.",
#     "A tiny bird sat on the edge of a branch, feathers puffed against the breeze. The morning sun painted the sky in soft colors. The bird tilted its head, then sang a clear, sweet note. The sound drifted far, carrying hope with it. Even small wings could fill the sky with song.",
#     "A squirrel scurried along a tree branch, tail flicking as it moved. It held an acorn tight, eyes quick and alert. The forest floor looked far below, but the squirrel did not fear. It jumped to the next branch with ease, carrying both food and courage high above the quiet ground.",
#     "A little hedgehog shuffled through the grass at dusk. Its tiny feet pressed soft trails in the earth. The air smelled of leaves and soil, cool against its nose. It paused when a breeze rustled, curling slightly before moving on. In the gentle half-light, the hedgehog searched for supper, steady and calm.",
# ]
text_sources = [
    "a tiger is crying and drink a beer"
]

def generate_speech(model, tokenizer, text_source, device, configs, generator, vocoder, lang=None, path2save_audio=None):
    spk = None
    tts_turn = configs['custom_data']['bos_tts'][lang]
    tts_eos = configs['custom_data']['eos_tts']
    print(f"{YELLOW}tts_turn: {tts_turn}, tts_eos: {tts_eos}{RESET}")
    text_source = remove_special_characters(text_source)
    encoded_inputs = tokenizer(text_source, return_tensors='pt', padding=True, truncation=True).to(device)
    encoded_inputs['input_ids'] = torch.cat(
        [encoded_inputs['input_ids'], torch.tensor([[tts_turn]], device='cuda:0')], dim=1)
    encoded_inputs['attention_mask'] = torch.cat(
        [encoded_inputs['attention_mask'], torch.tensor([[1]], device='cuda:0')], dim=1)
    encoded_inputs_length = encoded_inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model.speech_generate(
            encoded_inputs['input_ids'],
            max_new_tokens=config['inference']['max_length_unit'],
        )
        outputs = outputs.tolist()
        outputs = outputs[0][encoded_inputs_length:]
        # print(outputs)
        print(f"{'text source (GT)'.ljust(20)}: {text_source}")
        predict_unit = remove_consecutive_duplicates(outputs)
        x = torch.IntTensor(predict_unit).cuda()[None]
        x_lengths = torch.LongTensor([x.shape[-1]]).to(device)

        y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=configs["inference_gradtts"]["n_timesteps"],
                                               temperature=1.5,
                                               stoc=False, spk=spk, length_scale=0.91)
        audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
        if path2save_audio is not None:
            write(path2save_audio, 22050, audio)


def generate_image(model, tokenizer, vq_model, t5_model, text_source, device, configs, path2save_image=None):
    text_tokens_and_mask = tokenizer(
        text_source,
        max_length=120,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    text_tokens_and_mask['input_ids'] = text_tokens_and_mask['input_ids'].to(device)
    emb_masks = text_tokens_and_mask['attention_mask'].to(device)
    with torch.no_grad():
        text_encoder_embs = t5_model(
            input_ids=text_tokens_and_mask['input_ids'],
            attention_mask=emb_masks,
        )['last_hidden_state'].detach()  # [B, T, D], D = 2048

    new_emb_masks = torch.flip(emb_masks, dims=[-1])
    text_source_embs = []
    for i in range(text_encoder_embs.shape[0]):
        valid_num = int(new_emb_masks[i].sum().item())
        new_emb = torch.cat([text_encoder_embs[i, valid_num:], text_encoder_embs[i, :valid_num]])
        text_source_embs.append(new_emb)
    text_source_embs = torch.stack(text_source_embs)
    c_indices = text_source_embs * new_emb_masks[:, :, None]
    latent_size = 32
    codebook_embed_dim = 8
    qzshape = [len(c_indices), codebook_embed_dim, latent_size, latent_size]
    index_sample = model.image_generate(
        c_indices, latent_size**2,
        new_emb_masks, cfg_scale=7.5,
        temperature=1.0, top_k=1000,
        top_p=1.0, sample_logits=True,
    )
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    if path2save_image is not None:
        save_image(samples, path2save_image, normalize=True, value_range=(-1, 1))



def perform_inference(model, tokenizer, vq_model, t5_model, device, generator, vocoder, configs, lang=None):
    path2save_root = os.path.join(configs['training']['output_dir'], "samples")
    if not os.path.exists(path2save_root):
        os.makedirs(path2save_root)
    for index, text_source in enumerate(text_sources):
        path2save_audio = os.path.join(path2save_root, f"audio_{index}.wav")
        path2save_image = os.path.join(path2save_root, f"image_{index}.png")
        generate_speech(
            model=model,
            tokenizer=tokenizer,
            text_source=text_source,
            device=device,
            configs=configs,
            generator=generator,
            vocoder=vocoder,
            lang=lang,
            path2save_audio=path2save_audio
        )
        generate_image(
            model=model,
            tokenizer=tokenizer,
            t5_model=t5_model,
            vq_model=vq_model,
            text_source=text_source,
            device=device,
            configs=configs,
            path2save_image=path2save_image
        )

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

    codebook_size = 16384
    codebook_embed_dim = 8
    vq_ckpt = '/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/pretrained_models/vq_ds16_t2i.pt'
    vq_model = VQ_16(
        codebook_size=codebook_size,
        codebook_embed_dim=codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    vq_model.load_state_dict(torch.load(vq_ckpt, map_location="cpu")["model"])
    print(f"image tokenizer is loaded from {vq_ckpt}")
    # load t5 model
    t5_path = "/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/pretrained_models/t5-ckpt/flan-t5-xl"
    t5_model_kwargs = {'low_cpu_mem_usage': True, 'torch_dtype': torch.bfloat16}
    t5_model_kwargs['device_map'] = {'shared': device, 'encoder': device}
    t5_model = T5EncoderModel.from_pretrained(t5_path, **t5_model_kwargs).eval()
    print("T5 model loaded from:", t5_path)

    # 8. Load test data
    # path_file = "/home/ldap-users/s2220411/Code/new_explore_tts/MachineSpeechChain_ASRU25/dataset/LibriTTS/English/test-clean.json"
    path_dir = config['dataset']['path_dir']
    if len(path_dir) > 1:
        raise ValueError(f"Path directory should be a single path, got {path_dir}")
    else:
        path_dir = path_dir[0]

    lang = "English"  # Default language
    n_spks = 1
    generator = GradTTS(1001, n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    checkpoint = f"/home/ldap-users/s2220411/Code/new_explore_tts/Speech-Backbones/Grad-TTS/logs/CSS10_1K/{lang}/grad_250.pt"
    generator.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    perform_inference(model, tokenizer, vq_model, t5_model, device, generator, vocoder, configs=config, lang=lang)


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
