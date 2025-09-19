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
from tokenizer.tokenizer_image.vq_model import VQ_16
from transformers import T5EncoderModel, AutoTokenizer
from autoregressive.models.generate import generate as generate_img_fn
from torchvision.utils import save_image
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
    # distinct_numbers = [number_sequence[i] for i in range(len(number_sequence)) if i == 0 or number_sequence[i] != number_sequence[i - 1]]
    # remove token overrange 0-6560
    distinct_numbers = [x for x in number_sequence if int(x) < 6561]
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
        text = text.replace(char.strip(), " ")

    text = re.sub(_whitespace_re, ' ', text)
    return text.lower().strip()

image_prefix_promt = "Cartoon Style, No Artifacts, High Quality, No Text, No Noise, No Blur "
# text_sources = [
#     "A small rabbit hopped through the meadow after the rain. Its ears twitched at every sound, and its fur was still damp. It found a quiet spot under a leaf and watched drops fall to the ground. The world felt large and calm, and the rabbit simply listened, heart beating soft with wonder.",
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
"A brown dog barked happily in the sunny yard.",
"A fluffy white cat stretched lazily on the windowsill.",
"A pink piglet rolled playfully in the muddy puddle.",
"A black cow chewed grass slowly in the green pasture.",
"A proud rooster crowed loudly at sunrise.",
"A hen clucked softly as her chicks followed behind.",
"A white goose spread its wings beside the pond.",
"A duck splashed joyfully in the farmyard puddle.",
"A gray donkey brayed beside the wooden fence.",
"A brown horse galloped quickly across the open field.",
"A rabbit nibbled fresh carrots in the garden.",
"A sheep grazed quietly under the tall oak tree.",
"A goat jumped nimbly on the rocky hillside.",
"A turkey strutted proudly around the barnyard.",
"A pigeon pecked crumbs on the city street.",
"A guinea pig squeaked inside its little cage.",
"A hamster spun quickly in its round wheel.",
"A canary sang a cheerful song in the morning light.",
"A parakeet chirped brightly inside the living room.",
"A golden retriever chased a ball across the yard.",
"A black cat hid quietly under the chair.",
"A white rabbit hopped happily across the meadow.",
"A calico kitten played with a red ball of yarn.",
"A pony trotted playfully around the paddock.",
"A brown cowbell rang softly as the herd walked home.",
"A pair of ducks waddled along the village path.",
"A group of sheep huddled together in the shade.",
"A black goat munched grass near the stone wall.",
"A rooster flapped its wings proudly on the fence post.",
"A mother hen sheltered her chicks under her wings.",
"A spotted dog wagged its tail at the children.",
"A fluffy puppy licked the boy’s hand happily.",
"A ginger cat purred loudly on the soft cushion.",
"A white dove fluttered above the town square.",
"A small lamb followed closely behind its mother.",
"A brown ox pulled a heavy wooden cart.",
"A duckling paddled quickly behind its mother duck.",
"A horse neighed loudly in the village stable.",
"A farm dog ran circles around the flock of sheep.",
"A rabbit twitched its nose in the fresh grass.",
"A cat stretched its back in the warm sunlight.",
"A pig grunted happily while eating apples.",
"A foal trotted beside its mother horse.",
"A goose honked loudly as it crossed the path.",
"A chicken pecked grain from the ground.",
"A donkey carried baskets along the country road.",
"A rooster crowed again as the sun climbed higher.",
"A small terrier barked at the passing bicycle.",
"A cow drank water from the farm trough.",
"A white kitten chased a butterfly in the yard.",
"A golden lion roared across the wide savanna.",
"A gray elephant sprayed cool water with its trunk.",
"A red fox darted into the autumn forest leaves.",
"A tall giraffe reached for tender leaves in the tree.",
"A panda chewed bamboo quietly in the green valley.",
"A dolphin leapt high above the blue ocean waves.",
"A polar bear trudged slowly across the icy plain.",
"A peacock spread shimmering feathers under the sun.",
"A green turtle crawled across the sandy beach.",
"An owl hooted softly in the quiet night forest.",
"A monkey swung quickly between jungle branches.",
"A brown bear caught a fish in the rushing river.",
"A stag stood proudly at the forest edge.",
"A wolf padded silently through the snowy woods.",
"A raccoon washed food in the shallow stream.",
"A hedgehog curled up under the fallen leaves.",
"A camel rested under the hot desert sun.",
"A kangaroo hopped swiftly across the red sand.",
"A parrot squawked loudly in the tropical jungle.",
"A zebra grazed peacefully on the golden grassland.",
"A cheetah sprinted across the dry savanna plain.",
"A tiger prowled quietly in the thick jungle.",
"A crocodile basked lazily on the muddy bank.",
"A hippo yawned widely in the cool river.",
"A seal clapped its flippers near the rocky shore.",
"A penguin waddled slowly across the snowy ice.",
"A walrus rested with its tusks on the icy beach.",
"A whale spouted water high into the air.",
"A shark swam silently below the deep waves.",
"A flamingo stood gracefully in the shallow lagoon.",
"A lynx crept carefully through the snowy undergrowth.",
"A leopard stretched on a high tree branch.",
"A jaguar prowled near the forest river.",
"A gorilla beat its chest under the jungle canopy.",
"An orangutan reached for fruit in the tall tree.",
"A chimpanzee clapped loudly at sunset.",
"A sloth hung lazily from a tree branch.",
"An anteater searched the ground for ants.",
"A tapir waded slowly into the water.",
"A moose stood tall beside the mountain lake.",
"A reindeer trotted across the snowy field.",
"A bison grazed heavily on the wide prairie.",
"A yak climbed steadily along the rocky trail.",
"A panther crept quietly in the dark forest.",
"A hummingbird hovered above the red flower.",
"A toucan perched brightly on the rainforest branch.",
"A vulture circled slowly above the dry canyon.",
"A stork stood still in the marsh water.",
"A falcon dived swiftly toward its prey.",
"An eagle soared high over the mountain peaks.",
"A yellow school bus carried children down the morning road.",
"A red fire truck raced to the burning house.",
"A blue sailboat floated gently on the summer sea.",
"A silver airplane glided through the white clouds.",
"A long green train rolled through the mountain valley.",
"A tractor pulled heavy hay across the sunny field.",
"A bicycle rang its bell on the garden path.",
"A wooden boat rocked gently on the calm lake.",
"A black car cruised along the highway at night.",
"A hot air balloon drifted slowly into the sunrise sky.",
"A rocket blasted high into the night sky of stars.",
"A submarine slipped quietly under the ocean waves.",
"A scooter zipped along the busy city street.",
"A bright balloon floated above the crowded fairground.",
"A motorbike sped past the row of houses.",
"A taxi stopped at the busy corner of the square.",
"A bus picked up children in the pouring rain.",
"A train whistled loudly as it entered the station.",
"An airplane landed smoothly on the runway lights.",
"A delivery van parked outside the bakery shop.",
"A horse cart rattled along the village road.",
"A police car flashed blue lights in the dark street.",
"An ambulance rushed through traffic with sirens loud.",
"A garbage truck stopped to collect bins on the street.",
"A tram rolled slowly through the city center.",
"A jeep bounced across the rocky desert road.",
"A sports car sped quickly around the sharp corner.",
"A pickup truck carried hay bales to the farm.",
"A ferry carried cars across the wide river.",
"A fishing boat returned full of silver fish.",
"A cargo ship sailed steadily across the ocean.",
"A bulldozer pushed earth on the construction site.",
"A crane lifted heavy steel beams into place.",
"A fire engine ladder rose high against the building.",
"A snowplow cleared the road after the storm.",
"A tank rolled noisily across the training ground.",
"A race car zoomed around the circular track.",
"A helicopter hovered above the city skyline.",
"A glider floated silently in the summer sky.",
"A skateboard rolled down the hill with speed.",
"A kayak paddled quietly across the mountain lake.",
"A canoe drifted along the calm forest river.",
"A caravan parked by the beach for the night.",
"A delivery bike carried food through the traffic.",
"A trolleybus rumbled through the busy avenue.",
"A bulldozer cleared rubble from the old site.",
"A cement truck mixed concrete at the road works.",
"A metro train arrived at the underground station.",
"A rocket launched astronauts into the blue sky.",
"A teddy bear sat on the child’s bed waiting for a hug.",
"A spinning top twirled quickly on the wooden floor.",
"A paper kite flew high in the windy sky.",
"A red crayon drew a bright circle on the paper.",
"A storybook opened with pictures of castles and dragons.",
"A silver bell rang sweetly at the cottage door.",
"A lamp glowed warmly in the quiet room.",
"A piano played soft notes on the stage.",
"A trumpet shone brightly under the stage lights.",
"A drum thumped loudly in the parade.",
"A guitar rested against the old chair.",
"A violin sang softly in the orchestra.",
"A green apple lay fresh on the table.",
"An orange rolled across the kitchen counter.",
"A yellow banana curved gently in the basket.",
"A slice of watermelon dripped juice in the sun.",
"A notebook lay open with neat writing inside.",
"A pencil scribbled quickly on the page.",
"A rubber ball bounced across the playground.",
"A silver spoon clinked in the tea cup.",
"A pair of scissors snipped through the paper.",
"A ruler measured the length of the desk.",
"A school bag hung on the classroom hook.",
"A lunch box sat beside the bottle of water.",
"A toy train circled around the small track.",
"A toy car rolled under the sofa.",
"A doll wore a red dress with white shoes.",
"A teddy bear wore a ribbon around its neck.",
"A puzzle lay unfinished on the floor.",
"A stack of blocks toppled onto the carpet.",
"A balloon popped loudly in the air.",
"A ball of yarn rolled across the floor.",
"A football flew high into the net.",
"A basketball bounced on the court floor.",
"A tennis racket leaned against the wall.",
"A pair of skates glided over the ice rink.",
"A toy plane zoomed above the child’s head.",
"A toy boat floated in the bathtub water.",
"A bucket held sand beside the castle.",
"A spade dug into the soft sand.",
"A candle flickered gently on the table.",
"A clock ticked loudly on the wall.",
"A blanket folded neatly on the bed.",
"A chair stood quietly beside the desk.",
"A cushion rested on the sofa seat.",
"A mirror reflected the bright sunlight.",
"A vase of flowers stood on the window sill.",
"A broom leaned against the kitchen wall.",
"A basket of apples sat on the counter.",
"A rainbow stretched across the blue sky after rain.",
"The golden sun rose slowly above the quiet hills.",
"The silver moon shone brightly over the calm river.",
"A falling star streaked across the midnight sky.",
"A snowflake landed softly on the child’s mitten.",
"A warm breeze rustled through the green grass.",
"Spring rain tapped gently on the fresh leaves.",
"A golden autumn leaf drifted in the cool wind.",
"A flash of lightning lit the dark storm clouds.",
"A snowstorm covered the forest in white silence.",
"A tall pine tree stood strong on the mountain slope.",
"A cherry tree bloomed pink in the spring garden.",
"A sunflower turned its head toward the bright sun.",
"A lavender field spread purple across the hill.",
"A bamboo grove swayed gently in the morning air.",
"An orange pumpkin grew fat under the green leaves.",
"A willow tree bent its branches over the pond.",
"A cactus stood tall in the desert heat.",
"A wheat field waved golden in the summer wind.",
"A calm blue lake reflected the snowy peaks.",
"A river sparkled brightly in the sunshine.",
"A waterfall crashed loudly into the pool below.",
"The ocean waves rolled gently onto the shore.",
"A mountain peak glistened with snow in the sun.",
"A valley lay green between the tall hills.",
"A meadow bloomed with colorful wildflowers.",
"A desert stretched far under the blazing sun.",
"A storm cloud gathered dark in the sky.",
"A rainbow reappeared after the heavy rain.",
"A breeze carried the scent of jasmine flowers.",
"A thunderclap echoed across the valley.",
"A glacier shone white under the bright sun.",
"A volcano smoked slowly in the distance.",
"A canyon opened wide between the cliffs.",
"A coral reef glowed with fish underwater.",
"A tide rose high under the full moon.",
"A breeze shook the tall reeds by the river.",
"A maple tree turned red in the autumn air.",
"An oak tree spread its branches wide.",
"A pine forest whispered in the winter wind.",
"A lily floated on the still pond water.",
"A daisy bloomed white in the green meadow.",
"A rose opened red in the warm sun.",
"Tulips colored the garden in springtime.",
"A patch of clover spread across the field.",
"A snowdrift piled against the wooden fence.",
"A frost sparkled on the early morning grass.",
"The horizon glowed pink with sunset light.",
"The night sky filled with shining stars.",
"A dragonfly skimmed quickly over the still pond.",
"A lighthouse shone brightly across the stormy sea.",
"A campfire crackled warmly under the starry night sky.",
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
    fout_speech=None,
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

    # --- chuẩn hoá text
    # text_source = remove_special_characters(text_source)
    text_source = text_source.lower().replace("-", " ").strip()

    # --- tokenize + nối BOS_TTS (KHÔNG hardcode 'cuda:0')
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
        print(f"Saved: {path2save_audio}")

    transcript = transcription_whisper(path2save_audio, language=lang)
    print(f"{f'{RED}text source (GT)'.ljust(20)}: {remove_special_characters(text_source)}{RESET}")
    print(f"{f'{GREEN}transcript (ASR)'.ljust(20)}: {remove_special_characters(transcript)}{RESET}")
    out_file = f"{path2save_audio}\t{remove_special_characters(text_source)}\t{remove_special_characters(transcript)}\n"
    fout_speech.write(out_file)
    fout_speech.flush()

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



def perform_inference(model, tokenizer, vq_model, t5_model, device, frontend, cosyvoice_model, configs, lang=None):
    path2save_root = os.path.join(configs['training']['output_dir'], "samples_paper_realistic")
    if not os.path.exists(path2save_root):
        os.makedirs(path2save_root)
    fout_speech = open(os.path.join(path2save_root, "gen_speech.txt"), "w", encoding="utf-8")

    for index, text_source in enumerate(text_sources):
        path2save_audio = os.path.join(path2save_root, f"audio_{index}.wav")
        path2save_image = os.path.join(path2save_root, f"image_{index}.png")
        generate_speech(
            model=model,
            tokenizer=tokenizer,
            text_source=text_source,
            device=device,
            configs=configs,
            frontend=frontend,
            cosyvoice_model=cosyvoice_model,
            lang=lang,
            path2save_audio=path2save_audio,
            fout_speech=fout_speech
        )
        # generate_image(
        #     model=model,
        #     tokenizer=tokenizer,
        #     t5_model=t5_model,
        #     vq_model=vq_model,
        #     text_source= f"{image_prefix_promt} {text_source}",
        #     device=device,
        #     configs=configs,
        #     path2save_image=path2save_image
        # )

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

    ## load cosyvoice2
    model_dir = '/home/ldap-users/s2220411/Code/new_explore_tts/MachineSpeechChain_ASRU25/pretrained_models/CosyVoice2-0.5B'
    hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
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

    perform_inference(model, tokenizer, vq_model, t5_model, device, frontend, cosyvoice_model, configs=config, lang=lang)


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
