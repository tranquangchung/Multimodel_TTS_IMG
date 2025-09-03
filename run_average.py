#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
import pdb

import yaml
import torch
import glob
from autoregressive.models.gpt import GPT_XXL_speech, MultiTaskImageSpeech, GPT_XL
from collections import OrderedDict

def _load_sd(path):
    obj = os.path.join(path, "model.pth")
    print("Loading model from {}".format(obj))
    ckpt = torch.load(obj, map_location="cpu")
    return ckpt.get("model", ckpt)

def average_model_weights(models_or_paths):
    sds = [_load_sd(x) for x in models_or_paths]
    avg = OrderedDict()
    for k, v0 in sds[0].items():
        if torch.is_tensor(v0) and torch.is_floating_point(v0):
            stacked = torch.stack([sd[k].to(torch.float32) for sd in sds], 0)
            avg[k] = stacked.mean(0).to(v0.dtype)
        else:
            avg[k] = v0
    return avg

path_folder="/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce"
folder_name="LibriTTS_1e4_4Layer_16alpha_16rank_BS14_NoRemoveDup"
# Load models
model_names = glob.glob(f"{path_folder}/{folder_name}/checkpoint_iter_*")
# sort by iteration
model_names.sort(key=lambda x: int(x.split("_")[-1]))
# model_names = model_names[-3:]  # Load the last 3 models

avg_sd = average_model_weights(model_names)

out_path = os.path.join(path_folder, folder_name, "model_avg.pth")
torch.save({"model": avg_sd}, out_path)
print("Saved:", out_path)

