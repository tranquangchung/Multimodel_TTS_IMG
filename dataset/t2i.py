import os
import json
import pdb

import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class Text2ImgDatasetImg(Dataset):
    def __init__(self, lst_dir, face_lst_dir, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl
        for lst_name in sorted(os.listdir(lst_dir)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(lst_dir, lst_name)
            valid_file_path.append(file_path)
        
        # collect valid jsonl for face
        if face_lst_dir is not None:
            for lst_name in sorted(os.listdir(face_lst_dir)):
                if not lst_name.endswith('_face.jsonl'):
                    continue
                file_path = os.path.join(face_lst_dir, lst_name)
                valid_file_path.append(file_path)            
        
        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, code_name 


# class Text2ImgDataset(Dataset):
#     def __init__(self, args, transform):
#         img_path_list = []
#         valid_file_path = []
#         # collect valid jsonl file path
#         for lst_name in sorted(os.listdir(args.data_path)):
#             if not lst_name.endswith('.jsonl'):
#                 continue
#             file_path = os.path.join(args.data_path, lst_name)
#             valid_file_path.append(file_path)
#
#         for file_path in valid_file_path:
#             with open(file_path, 'r') as file:
#                 for line_idx, line in enumerate(file):
#                     data = json.loads(line)
#                     img_path = data['image_path']
#                     code_dir = file_path.split('/')[-1].split('.')[0]
#                     img_path_list.append((img_path, code_dir, line_idx))
#         self.img_path_list = img_path_list
#         self.transform = transform
#
#         self.t5_feat_path = args.t5_feat_path
#         self.short_t5_feat_path = args.short_t5_feat_path
#         self.t5_feat_path_base = self.t5_feat_path.split('/')[-1]
#         if self.short_t5_feat_path is not None:
#             self.short_t5_feat_path_base = self.short_t5_feat_path.split('/')[-1]
#         else:
#             self.short_t5_feat_path_base = self.t5_feat_path_base
#         self.image_size = args.image_size
#         latent_size = args.image_size // args.downsample_size
#         self.code_len = latent_size ** 2
#         self.t5_feature_max_len = 120
#         self.t5_feature_dim = 2048
#         self.max_seq_length = self.t5_feature_max_len + self.code_len
#
#     def __len__(self):
#         return len(self.img_path_list)
#
#     def dummy_data(self):
#         img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
#         t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
#         attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
#         valid = 0
#         return img, t5_feat_padding, attn_mask, valid
#
#     def __getitem__(self, index):
#         img_path, code_dir, code_name = self.img_path_list[index]
#         try:
#             img = Image.open(img_path).convert("RGB")
#         except:
#             img, t5_feat_padding, attn_mask, valid = self.dummy_data()
#             return img, t5_feat_padding, attn_mask, torch.tensor(valid)
#
#         if min(img.size) < self.image_size:
#             img, t5_feat_padding, attn_mask, valid = self.dummy_data()
#             return img, t5_feat_padding, attn_mask, torch.tensor(valid)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         t5_file = os.path.join(self.t5_feat_path, code_dir, f"{code_name}.npy")
#         if torch.rand(1) < 0.3:
#             t5_file = t5_file.replace(self.t5_feat_path_base, self.short_t5_feat_path_base)
#
#         t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
#         if os.path.isfile(t5_file):
#             try:
#                 t5_feat = torch.from_numpy(np.load(t5_file))
#                 t5_feat_len = t5_feat.shape[1]
#                 feat_len = min(self.t5_feature_max_len, t5_feat_len)
#                 t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
#                 emb_mask = torch.zeros((self.t5_feature_max_len,))
#                 emb_mask[-feat_len:] = 1
#                 attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
#                 T = self.t5_feature_max_len
#                 attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
#                 eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
#                 attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
#                 attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
#                 valid = 1
#             except:
#                 img, t5_feat_padding, attn_mask, valid = self.dummy_data()
#         else:
#             img, t5_feat_padding, attn_mask, valid = self.dummy_data()
#
#         return img, t5_feat_padding, attn_mask, torch.tensor(valid)

class Text2ImgDataset(Dataset):
    def __init__(self, args, transform):
        img_path_list = []
        valid_file_path = []
        self.img_path_list = img_path_list
        self.transform = transform

        self.t5_feat_path = args.t5_feat_path
        self.short_t5_feat_path = args.short_t5_feat_path
        self.t5_feat_path_base = self.t5_feat_path.split('/')[-1]
        if self.short_t5_feat_path is not None:
            self.short_t5_feat_path_base = self.short_t5_feat_path.split('/')[-1]
        else:
            self.short_t5_feat_path_base = self.t5_feat_path_base
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

    def __len__(self):
        return 1000

    # def dummy_data(self):
    #     # Random image: 3 x H x W, normalized-like float tensor
    #     # img = torch.randn((3, self.image_size, self.image_size), dtype=torch.float32)
    #     # T = random.randint(200, 600)
    #     T = 1000
    #     img = torch.randint(0, 16328, (T,), dtype=torch.long) # speech token
    #
    #     # Random T5 features: 1 x max_len x feature_dim (e.g., 1 x 120 x 2048)
    #     t5_feat_padding = torch.randn((1, self.t5_feature_max_len, self.t5_feature_dim))
    #
    #     # Causal attention mask (with lower triangle), batch dimension added
    #     max_seq_length = T + self.t5_feature_max_len
    #     attn_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool)).unsqueeze(0)
    #
    #     # Valid flag: set to 1 to simulate "valid" sample
    #     valid = torch.tensor(1, dtype=torch.int64)
    #     return img, t5_feat_padding, attn_mask, valid

    def dummy_data(self):
        # Random sequence length per sample
        T = torch.randint(100, 600, ()).item()  # e.g., 256, 512, 800
        # T = 600
        # Speech tokens: [T]
        img = torch.randint(0, 16384, (T,), dtype=torch.long)

        # T5 features: [120, 2048]
        t5_feat = torch.randn(self.t5_feature_max_len, self.t5_feature_dim, dtype=torch.float32)

        # Valid flag (scalar)
        valid = torch.tensor(1, dtype=torch.int64)

        return img, t5_feat, valid

    def text2speech_collate_fn(self, batch):
        img_tokens_list = []
        t5_feat_list = []
        valid_list = []
        T_list = []

        for img_tokens, t5_feat, valid in batch:
            img_tokens_list.append(img_tokens)
            t5_feat_list.append(t5_feat)
            valid_list.append(valid)
            T_list.append(img_tokens.shape[0])
        B = len(batch)
        T_max = max(T_list)
        S_max = 120 + T_max  # cls_token_num + max token length
        # Pad image tokens
        img_tokens_padded = torch.zeros(B, T_max, dtype=torch.long)
        for i, tokens in enumerate(img_tokens_list):
            T = tokens.shape[0]
            img_tokens_padded[i, :T] = tokens
        attn_mask_padded = []
        for i, T in enumerate(T_list):
            S = self.t5_feature_max_len
            seq_length = S + T
            attn_mask = [1]*seq_length + [0]*(S_max - seq_length)
            attn_mask_padded.append(attn_mask)
        attn_mask_padded = torch.tensor(attn_mask_padded, dtype=torch.bool)
        cache_position = torch.arange(S_max, device=attn_mask_padded.device)
        casual_mask_padded = self._prepare_4d_causal_attention(
            attn_mask_padded,
            sequence_length=S_max,
            target_length=S_max,
            batch_size=B,
            cache_position=cache_position
        )

        t5_feat_batch = torch.stack(t5_feat_list)  # [B, 120, 2048]
        valid_batch = torch.stack(valid_list)
        return img_tokens_padded, t5_feat_batch, casual_mask_padded, valid_batch

    def __getitem__(self, index):
        # Always return dummy data regardless of index or actual data
        return self.dummy_data()

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

        return causal_mask



class Text2ImgDatasetCode(Dataset):
    def __init__(self, args):
        pass




def build_t2i_image(args, transform):
    return Text2ImgDatasetImg(args.data_path, args.data_face_path, transform)

def build_t2i(args, transform):
    return Text2ImgDataset(args, transform)

def build_t2i_code(args):
    return Text2ImgDatasetCode(args)