
import torch
from torch.utils.data import Dataset
import pdb
import librosa
import sys
sys.path.append('/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen')
import utils_text
from utils_text import remove_duplicates, plot_histogram, remove_duplicates
import numpy as np
import termplotlib as tpl
import json
import glob
import os.path
import random
import numtext as nt
import re
_whitespace_re = re.compile(r'\s+')
from .voice_instruction_builder_V2 import instruction_from_filename
from numpy import random


def plot_histogram(data, type_data="UNIT"):
    if type_data == "UNIT":
        color = utils_text.GREEN
    elif type_data == "TEXT":
        color = utils_text.RED
    print(f"{color}Data with {type_data} length distribution")
    hist, bins = np.histogram(data, bins=20)
    # Create and show the plot
    fig = tpl.figure()
    fig.hist(hist, bins, force_ascii=True, orientation="horizontal")
    fig.show()
    print(f"{utils_text.RESET}")

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

class DatasetT2S(Dataset):
    def __init__(self, tokenizer, split, config, args=None):
        self.config = config
        self.split = split
        path_dir_list = config['dataset']['path_dir']
        langs = config['dataset']['langs']
        self.tokenizer = tokenizer
        if args.debug:
            train_samples = config['debug']['train_samples']
            dev_samples = config['debug']['dev_samples']
        else:
            train_samples = config['dataset']['train_samples']
            dev_samples = config['dataset']['dev_samples']
        data_list = []
        training_files = config['dataset']['train_files']
        dev_files = config['dataset']['dev_files']
        if split == 'train':
            for path_dir in path_dir_list:
                for lang in langs:
                    for file in training_files:
                        path_file = os.path.join(path_dir, lang, file)
                        if os.path.exists(path_file):
                            print(f"Loading {path_file}")
                            with open(path_file, 'r') as f:
                                data = json.load(f)[:train_samples]
                                print(f"{lang} - {file}: {len(data)}")
                                data_list += data
        elif split == 'dev':
            for path_dir in path_dir_list:
                for lang in langs:
                    for file in dev_files:
                        path_file = os.path.join(path_dir, lang, file)
                        if os.path.exists(path_file):
                            print(f"Loading {path_file}")
                            with open(path_file, 'r') as f:
                                data = json.load(f)[:dev_samples]
                                print(f"{lang}: {len(data)}")
                                data_list += data
        self.data, unit_list_lens, text_list_lens = self._filter_data(data_list)

        if split == 'train' or split == 'dev':
            plot_histogram(unit_list_lens, type_data="UNIT")
            plot_histogram(text_list_lens, type_data="TEXT")
        print(f"Total data: {len(self.data)}")

        self.languages_mapping = {
            "Dutch": "nl_XX",
            "French": "fr_XX",
            "German": "de_DE",
            "Italian": "it_IT",
            "Polish": "pl_PL",
            "Portuguese": "pt_PT",
            "Spanish": "es_XX",
            "English": "en_XX",
        }
        # reverse mapping
        self.languages_mapping_rev = {v: k for k, v in self.languages_mapping.items()}

        # reverse mapping
        self.ignore_token = self.config['custom_data']['ignore_token']
        self.bos_tts = self.config['custom_data']['bos_tts']
        self.eos_tts = self.config['custom_data']['eos_tts']
        self.bos_asr = self.config['custom_data']['bos_asr']
        self.eos_asr = self.config['custom_data']['eos_asr']
        self.model_max_length = 200

    def __len__(self):
        return len(self.data)

    def processing_label(self, tokens, remove_consecutive=False):
        processed_tokens = remove_duplicates(tokens, remove_consecutive=False)
        # Convert list of strings to int
        for i in range(len(processed_tokens)):
            processed_tokens[i] = int(processed_tokens[i])
        return processed_tokens


    def _filter_data(self, data):
        list_filter = []
        unit_list_lens = []
        text_list_lens = []
        for item in data:
            keep_text = False
            keep_unit = False
            target_unit = item["unit"]
            target_unit = remove_duplicates(target_unit)
            len_target_unit = len(target_unit)
            try:
                len_text_unit = len(item["transcript"].split(' '))
                if 3 < len_text_unit < self.config['preprocessing']['max_length_text']:
                    keep_text = True
                if 10 < len_target_unit < self.config['preprocessing']['max_length_unit']:
                    keep_unit = True
                if keep_text and keep_unit:
                    list_filter.append(item)
                    unit_list_lens.append(len_target_unit)
                    text_list_lens.append(len_text_unit)
            except Exception as e:
                print(f"Error processing item: {item}, Error: {e}")
                continue
        return list_filter, unit_list_lens, text_list_lens

    def clean_outer_quotes(self, s: str) -> str:
        s = s.strip()
        # nếu chuỗi bao ngoài toàn bộ bằng 1 cặp '...' hoặc "..." thì bỏ và dùng "
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            # bỏ cặp ngoài cùng và đổi thành dùng dấu "
            inner = s[1:-1].strip()
            return f'"{inner}"'  # ép bao ngoài thành nháy kép
        # nếu toàn bộ chuỗi chỉ là một dấu '
        if s == "'":
            return '"'
        return s

    def __getitem__(self, idx):
        item = self.data[idx]
        # source_text = remove_special_characters(item['transcript'])
        source_text = item['transcript']
        target_unit = item['unit']
        language_source = item['language']
        model_inputs = self.tokenizer(
            source_text,
            max_length=self.model_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        file_name = item.get('audio_path').split('/')[-1].replace('.wav', '')
        # prompt_instruction = instruction_from_filename(file_name, n_per_file=6)
        # print(f"{file_name} -- {prompt_instruction}")
        prompt_instruction = ''
        if 'text_description' in item:
            prompt_instruction = item['text_description']


        text_token_len = model_inputs["attention_mask"].sum().item()
        text_task = model_inputs["input_ids"][0][:text_token_len].tolist()
        unit_task = self.processing_label(target_unit, remove_consecutive=True)

        # TTS (text -> unit)
        bos_tts = self.bos_tts[self.languages_mapping_rev[language_source]]
        input_ids_tts = text_task + [bos_tts] + unit_task + [self.eos_tts]
        attention_mask_tts = [1] * len(input_ids_tts)
        labels_tts = [self.ignore_token] * len(text_task) + [bos_tts] + unit_task + [self.eos_tts]
        len_token_input_tts = len(input_ids_tts)
        len_token_label_tts = len(labels_tts)

        # ASR (unit -> text)
        bos_asr = self.bos_asr[self.languages_mapping_rev[language_source]]
        input_ids_asr = unit_task + [bos_asr] + text_task + [self.eos_asr]
        attention_mask_asr = [1] * len(input_ids_asr)
        labels_asr = [self.ignore_token] * len(unit_task) + [bos_asr] + text_task + [self.eos_asr]
        len_token_input_asr = len(input_ids_asr)
        len_token_label_asr = len(labels_asr)

        # Return both TTS and ASR data
        return {
            'input_ids_tts': input_ids_tts,
            'attention_mask_tts': attention_mask_tts,
            'len_token_input_tts': len_token_input_tts,
            'labels_tts': labels_tts,
            'len_token_label_tts': len_token_label_tts,

            'input_ids_asr': input_ids_asr,
            'attention_mask_asr': attention_mask_asr,
            'len_token_input_asr': len_token_input_asr,
            'labels_asr': labels_asr,
            'len_token_label_asr': len_token_label_asr,

            'language_source': language_source,
            'text_source': source_text,
            'prompt_instruction': prompt_instruction,
        }

    def collate_fn(self, batch):
        # Sort the batch based on the length of the input (TTS and ASR separately)
        batch = sorted(batch, key=lambda x: x['len_token_input_tts'], reverse=True)

        # Extract components for TTS
        input_ids_tts = [item['input_ids_tts'] for item in batch]
        attention_mask_tts = [item['attention_mask_tts'] for item in batch]
        len_token_input_tts = [item['len_token_input_tts'] for item in batch]
        labels_tts = [item['labels_tts'] for item in batch]
        len_token_label_tts = [item['len_token_label_tts'] for item in batch]

        # Extract components for ASR
        input_ids_asr = [item['input_ids_asr'] for item in batch]
        attention_mask_asr = [item['attention_mask_asr'] for item in batch]
        len_token_input_asr = [item['len_token_input_asr'] for item in batch]
        labels_asr = [item['labels_asr'] for item in batch]
        len_token_label_asr = [item['len_token_label_asr'] for item in batch]

        prompt_instructions = [item['prompt_instruction'] for item in batch]

        # Padding for TTS
        max_len_tts = max([len(ids) for ids in input_ids_tts])
        input_ids_tts_padded = []
        attention_mask_tts_padded = []
        labels_tts_padded = []

        for i in range(len(input_ids_tts)):
            input_ids_tts_padded.append(
                input_ids_tts[i] + [self.config["custom_data"]["padding_eos"]] * (max_len_tts - len(input_ids_tts[i])))  # Pad with a dummy token
            attention_mask_tts_padded.append(attention_mask_tts[i] + [0] * (max_len_tts - len(attention_mask_tts[i])))
            labels_tts_padded.append(
                labels_tts[i] + [-100] * (max_len_tts - len(labels_tts[i])))  # Pad with -100 for ignored tokens

        # Padding for ASR
        max_len_asr = max([len(ids) for ids in input_ids_asr])
        input_ids_asr_padded = []
        attention_mask_asr_padded = []
        labels_asr_padded = []

        for i in range(len(input_ids_asr)):
            input_ids_asr_padded.append(
                input_ids_asr[i] + [self.config["custom_data"]["padding_eos"]] * (max_len_asr - len(input_ids_asr[i])))  # Pad with a dummy token
            attention_mask_asr_padded.append(attention_mask_asr[i] + [0] * (max_len_asr - len(attention_mask_asr[i])))
            labels_asr_padded.append(
                labels_asr[i] + [-100] * (max_len_asr - len(labels_asr[i])))  # Pad with -100 for ignored tokens

        # Convert lists to tensors
        input_ids_tts_padded = torch.LongTensor(input_ids_tts_padded)
        attention_mask_tts_padded = torch.LongTensor(attention_mask_tts_padded)
        attention_mask_tts_padded = self._prepare_4d_causal_attention(
            attention_mask_tts_padded,
            sequence_length=input_ids_tts_padded.shape[1],
            target_length=input_ids_tts_padded.shape[1],
            batch_size=input_ids_tts_padded.shape[0],
            cache_position=torch.arange(input_ids_tts_padded.shape[1], device=input_ids_tts_padded.device, dtype=torch.long)
        )
        labels_tts_padded = torch.LongTensor(labels_tts_padded)
        input_ids_asr_padded = torch.LongTensor(input_ids_asr_padded)
        attention_mask_asr_padded = torch.LongTensor(attention_mask_asr_padded)
        attention_mask_asr_padded = self._prepare_4d_causal_attention(
            attention_mask_asr_padded,
            sequence_length=input_ids_asr_padded.shape[1],
            target_length=input_ids_asr_padded.shape[1],
            batch_size=input_ids_asr_padded.shape[0],
            cache_position=torch.arange(input_ids_asr_padded.shape[1], device=input_ids_asr_padded.device, dtype=torch.long)
        )

        labels_asr_padded = torch.LongTensor(labels_asr_padded)

        len_token_input_tts = torch.LongTensor(len_token_input_tts)
        len_token_label_tts = torch.LongTensor(len_token_label_tts)

        len_token_input_asr = torch.LongTensor(len_token_input_asr)
        len_token_label_asr = torch.LongTensor(len_token_label_asr)

        # Extract other metadata
        language_source = [item['language_source'] for item in batch]
        text_source = [item['text_source'] for item in batch]

        # Return the padded batch
        return {
            'input_ids_tts': input_ids_tts_padded,
            'attention_mask_tts': attention_mask_tts_padded,
            'labels_tts': labels_tts_padded,
            'len_token_input_tts': len_token_input_tts,
            'len_token_label_tts': len_token_label_tts,

            'input_ids_asr': input_ids_asr_padded,
            'attention_mask_asr': attention_mask_asr_padded,
            'labels_asr': labels_asr_padded,
            'len_token_input_asr': len_token_input_asr,
            'len_token_label_asr': len_token_label_asr,

            'language_source': language_source,
            'text_source': text_source,
            'prompt_instructions': prompt_instructions,
        }

    def _prepare_4d_causal_attention(
            self,
            attention_mask,
            sequence_length: int,
            target_length: int,
            batch_size: int,
            cache_position = None,
    ):
        device = attention_mask.device if attention_mask is not None else cache_position.device
        dtype = attention_mask.dtype if attention_mask is not None else torch.float32
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = 1 #torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        invert = 1 - causal_mask
        return invert.to(dtype=torch.bool)

class DatasetT2S_Prompt(Dataset):
    def __init__(self, tokenizer, split, config, args=None):
        self.config = config
        self.split = split
        path_dir_list = config['dataset']['path_dir']
        langs = config['dataset']['langs']
        self.tokenizer = tokenizer
        if args.debug:
            train_samples = config['debug']['train_samples']
            dev_samples = config['debug']['dev_samples']
        else:
            train_samples = config['dataset']['train_samples']
            dev_samples = config['dataset']['dev_samples']
        data_list = []
        training_files = config['dataset']['train_files']
        dev_files = config['dataset']['dev_files']
        if split == 'train':
            for path_dir in path_dir_list:
                for lang in langs:
                    for file in training_files:
                        path_file = os.path.join(path_dir, lang, file)
                        if os.path.exists(path_file):
                            print(f"Loading {path_file}")
                            with open(path_file, 'r') as f:
                                data = json.load(f)[:train_samples]
                                print(f"{lang} - {file}: {len(data)}")
                                data_list += data
        elif split == 'dev':
            for path_dir in path_dir_list:
                for lang in langs:
                    for file in dev_files:
                        path_file = os.path.join(path_dir, lang, file)
                        if os.path.exists(path_file):
                            print(f"Loading {path_file}")
                            with open(path_file, 'r') as f:
                                data = json.load(f)[:dev_samples]
                                print(f"{lang}: {len(data)}")
                                data_list += data
        self.data, unit_list_lens, text_list_lens = self._filter_data(data_list)

        if split == 'train' or split == 'dev':
            plot_histogram(unit_list_lens, type_data="UNIT")
            plot_histogram(text_list_lens, type_data="TEXT")
        print(f"Total data: {len(self.data)}")

        self.languages_mapping = {
            "Dutch": "nl_XX",
            "French": "fr_XX",
            "German": "de_DE",
            "Italian": "it_IT",
            "Polish": "pl_PL",
            "Portuguese": "pt_PT",
            "Spanish": "es_XX",
            "English": "en_XX",
        }
        # reverse mapping
        self.languages_mapping_rev = {v: k for k, v in self.languages_mapping.items()}

        # reverse mapping
        self.ignore_token = self.config['custom_data']['ignore_token']
        self.bos_tts = self.config['custom_data']['bos_tts']
        self.eos_tts = self.config['custom_data']['eos_tts']
        self.bos_asr = self.config['custom_data']['bos_asr']
        self.eos_asr = self.config['custom_data']['eos_asr']
        self.model_max_length = 200

    def __len__(self):
        return len(self.data)

    def processing_label(self, tokens, remove_consecutive=False):
        processed_tokens = remove_duplicates(tokens, remove_consecutive=False)
        # Convert list of strings to int
        for i in range(len(processed_tokens)):
            processed_tokens[i] = int(processed_tokens[i])
        return processed_tokens


    def _filter_data(self, data):
        list_filter = []
        unit_list_lens = []
        text_list_lens = []
        for item in data:
            keep_text = False
            keep_unit = False
            target_unit = item["unit"]
            target_unit = remove_duplicates(target_unit)
            len_target_unit = len(target_unit)
            try:
                len_text_unit = len(item["transcript"].split(' '))
                if 3 < len_text_unit < self.config['preprocessing']['max_length_text']:
                    keep_text = True
                if 10 < len_target_unit < self.config['preprocessing']['max_length_unit']:
                    keep_unit = True
                if keep_text and keep_unit:
                    list_filter.append(item)
                    unit_list_lens.append(len_target_unit)
                    text_list_lens.append(len_text_unit)
            except Exception as e:
                print(f"Error processing item: {item}, Error: {e}")
                continue
        return list_filter, unit_list_lens, text_list_lens

    def clean_outer_quotes(self, s: str) -> str:
        s = s.strip()
        # nếu chuỗi bao ngoài toàn bộ bằng 1 cặp '...' hoặc "..." thì bỏ và dùng "
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            # bỏ cặp ngoài cùng và đổi thành dùng dấu "
            inner = s[1:-1].strip()
            return f'"{inner}"'  # ép bao ngoài thành nháy kép
        # nếu toàn bộ chuỗi chỉ là một dấu '
        if s == "'":
            return '"'
        return s

    def __getitem__(self, idx):
        item = self.data[idx]
        # source_text = remove_special_characters(item['transcript'])
        source_text = item['transcript']
        target_unit = item['unit']
        language_source = item['language']

        file_name = item.get('audio_path').split('/')[-1].replace('.wav', '')
        # prompt_instruction = instruction_from_filename(file_name, n_per_file=6)
        # print(f"{file_name} -- {prompt_instruction}")
        prompt_instruction = item['text_description'].replace("'", "").replace('"', "")
        source_text = f"<instruct>{prompt_instruction}</instruct>. {source_text}"
        # source_text = f"<instruct>{prompt_instruction}</instruct>. {prompt_instruction}"

        model_inputs = self.tokenizer(
            source_text,
            max_length=self.model_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        text_token_len = model_inputs["attention_mask"].sum().item()
        text_task = model_inputs["input_ids"][0][:text_token_len].tolist()
        unit_task = self.processing_label(target_unit, remove_consecutive=True)
        # TTS (text -> unit)
        bos_tts = self.bos_tts[self.languages_mapping_rev[language_source]]
        input_ids_tts = text_task + [bos_tts] + unit_task + [self.eos_tts]
        attention_mask_tts = [1] * len(input_ids_tts)
        labels_tts = [self.ignore_token] * len(text_task) + [bos_tts] + unit_task + [self.eos_tts]
        len_token_input_tts = len(input_ids_tts)
        len_token_label_tts = len(labels_tts)

        # ASR (unit -> text)
        bos_asr = self.bos_asr[self.languages_mapping_rev[language_source]]
        input_ids_asr = unit_task + [bos_asr] + text_task + [self.eos_asr]
        attention_mask_asr = [1] * len(input_ids_asr)
        labels_asr = [self.ignore_token] * len(unit_task) + [bos_asr] + text_task + [self.eos_asr]
        len_token_input_asr = len(input_ids_asr)
        len_token_label_asr = len(labels_asr)


        # Return both TTS and ASR data
        return {
            'input_ids_tts': input_ids_tts,
            'attention_mask_tts': attention_mask_tts,
            'len_token_input_tts': len_token_input_tts,
            'labels_tts': labels_tts,
            'len_token_label_tts': len_token_label_tts,

            'input_ids_asr': input_ids_asr,
            'attention_mask_asr': attention_mask_asr,
            'len_token_input_asr': len_token_input_asr,
            'labels_asr': labels_asr,
            'len_token_label_asr': len_token_label_asr,

            'language_source': language_source,
            'text_source': source_text,
            'prompt_instruction': prompt_instruction,
        }

    def collate_fn(self, batch):
        # Sort the batch based on the length of the input (TTS and ASR separately)
        batch = sorted(batch, key=lambda x: x['len_token_input_tts'], reverse=True)

        # Extract components for TTS
        input_ids_tts = [item['input_ids_tts'] for item in batch]
        attention_mask_tts = [item['attention_mask_tts'] for item in batch]
        len_token_input_tts = [item['len_token_input_tts'] for item in batch]
        labels_tts = [item['labels_tts'] for item in batch]
        len_token_label_tts = [item['len_token_label_tts'] for item in batch]

        # Extract components for ASR
        input_ids_asr = [item['input_ids_asr'] for item in batch]
        attention_mask_asr = [item['attention_mask_asr'] for item in batch]
        len_token_input_asr = [item['len_token_input_asr'] for item in batch]
        labels_asr = [item['labels_asr'] for item in batch]
        len_token_label_asr = [item['len_token_label_asr'] for item in batch]

        prompt_instructions = [item['prompt_instruction'] for item in batch]

        # Padding for TTS
        max_len_tts = max([len(ids) for ids in input_ids_tts])
        input_ids_tts_padded = []
        attention_mask_tts_padded = []
        labels_tts_padded = []

        for i in range(len(input_ids_tts)):
            input_ids_tts_padded.append(
                input_ids_tts[i] + [self.config["custom_data"]["padding_eos"]] * (max_len_tts - len(input_ids_tts[i])))  # Pad with a dummy token
            attention_mask_tts_padded.append(attention_mask_tts[i] + [0] * (max_len_tts - len(attention_mask_tts[i])))
            labels_tts_padded.append(
                labels_tts[i] + [-100] * (max_len_tts - len(labels_tts[i])))  # Pad with -100 for ignored tokens

        # Padding for ASR
        max_len_asr = max([len(ids) for ids in input_ids_asr])
        input_ids_asr_padded = []
        attention_mask_asr_padded = []
        labels_asr_padded = []

        for i in range(len(input_ids_asr)):
            input_ids_asr_padded.append(
                input_ids_asr[i] + [self.config["custom_data"]["padding_eos"]] * (max_len_asr - len(input_ids_asr[i])))  # Pad with a dummy token
            attention_mask_asr_padded.append(attention_mask_asr[i] + [0] * (max_len_asr - len(attention_mask_asr[i])))
            labels_asr_padded.append(
                labels_asr[i] + [-100] * (max_len_asr - len(labels_asr[i])))  # Pad with -100 for ignored tokens

        # Convert lists to tensors
        input_ids_tts_padded = torch.LongTensor(input_ids_tts_padded)
        attention_mask_tts_padded = torch.LongTensor(attention_mask_tts_padded)
        attention_mask_tts_padded = self._prepare_4d_causal_attention(
            attention_mask_tts_padded,
            sequence_length=input_ids_tts_padded.shape[1],
            target_length=input_ids_tts_padded.shape[1],
            batch_size=input_ids_tts_padded.shape[0],
            cache_position=torch.arange(input_ids_tts_padded.shape[1], device=input_ids_tts_padded.device, dtype=torch.long)
        )
        labels_tts_padded = torch.LongTensor(labels_tts_padded)
        input_ids_asr_padded = torch.LongTensor(input_ids_asr_padded)
        attention_mask_asr_padded = torch.LongTensor(attention_mask_asr_padded)
        attention_mask_asr_padded = self._prepare_4d_causal_attention(
            attention_mask_asr_padded,
            sequence_length=input_ids_asr_padded.shape[1],
            target_length=input_ids_asr_padded.shape[1],
            batch_size=input_ids_asr_padded.shape[0],
            cache_position=torch.arange(input_ids_asr_padded.shape[1], device=input_ids_asr_padded.device, dtype=torch.long)
        )

        labels_asr_padded = torch.LongTensor(labels_asr_padded)

        len_token_input_tts = torch.LongTensor(len_token_input_tts)
        len_token_label_tts = torch.LongTensor(len_token_label_tts)

        len_token_input_asr = torch.LongTensor(len_token_input_asr)
        len_token_label_asr = torch.LongTensor(len_token_label_asr)

        # Extract other metadata
        language_source = [item['language_source'] for item in batch]
        text_source = [item['text_source'] for item in batch]

        # Return the padded batch
        return {
            'input_ids_tts': input_ids_tts_padded,
            'attention_mask_tts': attention_mask_tts_padded,
            'labels_tts': labels_tts_padded,
            'len_token_input_tts': len_token_input_tts,
            'len_token_label_tts': len_token_label_tts,

            'input_ids_asr': input_ids_asr_padded,
            'attention_mask_asr': attention_mask_asr_padded,
            'labels_asr': labels_asr_padded,
            'len_token_input_asr': len_token_input_asr,
            'len_token_label_asr': len_token_label_asr,

            'language_source': language_source,
            'text_source': text_source,
            'prompt_instructions': prompt_instructions,
        }

    def _prepare_4d_causal_attention(
            self,
            attention_mask,
            sequence_length: int,
            target_length: int,
            batch_size: int,
            cache_position = None,
    ):
        device = attention_mask.device if attention_mask is not None else cache_position.device
        dtype = attention_mask.dtype if attention_mask is not None else torch.float32
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = 1 #torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        invert = 1 - causal_mask
        return invert.to(dtype=torch.bool) # True 👉 ĐƯỢC attend, False 👉 CHẶN.

    # def _prepare_4d_causal_attention(
    #     self,
    #     attention_mask,              # [B, L] với 1=valid, 0=pad; hoặc None
    #     sequence_length: int,
    #     target_length: int,
    #     batch_size: int,
    #     cache_position=None,         # không dùng cho training full seq; có thể bỏ qua
    # ):
    #     # Lấy device
    #     device = attention_mask.device if attention_mask is not None else (
    #         cache_position.device if cache_position is not None else torch.device("cpu")
    #     )
    #
    #     # ----- 1) Causal mask (True = MASKED) -----
    #     # che tam giác trên: tại truy vấn i, không được nhìn j>i
    #     Lq = target_length
    #     Lk = sequence_length
    #     causal = torch.triu(torch.ones(Lq, Lk, dtype=torch.bool, device=device), diagonal=1)  # [Lq, Lk]
    #
    #     # ----- 2) Padding mask (True = MASKED) -----
    #     if attention_mask is not None:
    #         # attention_mask: [B, Lk] với 1=keep, 0=pad -> True ở pad
    #         pad = ~attention_mask.bool()                              # [B, Lk]
    #         pad4d = pad[:, None, None, :].expand(batch_size, 1, Lq, Lk)
    #     else:
    #         pad4d = torch.zeros((batch_size, 1, Lq, Lk), dtype=torch.bool, device=device)
    #
    #     # ----- 3) Final 4D mask -----
    #     mask4d = causal[None, None, :, :].expand(batch_size, 1, Lq, Lk) | pad4d  # True=masked
    #     return mask4d