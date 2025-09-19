#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pdb
import shutil
from typing import Optional, Tuple, List
import jiwer

# ====== ĐƯỜNG DẪN ======
ROOT_PATH = "/home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen"
DATASET_TXT = "/home/ldap-users/quangchung-t/Code/new_explore_multimodel/LlamaGen/result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce/LibriTTS_1e4_8Layer_16alpha_16rank_BS14_RemoveDup_KeepPunctuation/samples_paper_realistic/gen_speech.txt"
DEST_DIR = "/home/ldap-users/quangchung-t/Code/new_explore_multimodel/LlamaGen/result/TTS_result/ImageSpeechGeneration_Final_Cosyvoce/LibriTTS_1e4_8Layer_16alpha_16rank_BS14_RemoveDup_KeepPunctuation/samples_paper_realistic_filter"

# ====== TRANSFORMS ======
WER_TX = jiwer.Compose([
    jiwer.Strip(), jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords()
])
CER_TX = jiwer.Compose([
    jiwer.Strip(), jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfChars()
])

fmt = lambda x: f"{x:.2%}"

def parse_line(line: str) -> Optional[Tuple[str, str, str]]:
    parts = line.rstrip("\n").split("\t")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None

def empty_after(tx: jiwer.Compose, s: str) -> bool:
    try:
        out = tx(s)
        return not out or not out[0]
    except:
        return True

def guess_image_path(audio_path: str) -> str:
    base = os.path.basename(audio_path)
    name, _ = os.path.splitext(base)
    if "audio_" in name:
        new_name = name.replace("audio_", "image_", 1) + ".png"
    else:
        new_name = name.replace("audio", "image") + ".png"
    return os.path.join(os.path.dirname(audio_path), new_name)

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    name_text_path = os.path.join(DEST_DIR, "dataset.txt")

    items: List[dict] = []

    # Lọc các dòng có WER < 5%
    with open(DATASET_TXT, encoding="utf-8") as fin:
        for i, line in enumerate(fin, 1):
            parsed = parse_line(line)
            if not parsed:
                continue
            path_audio, gt, pred = parsed
            if empty_after(WER_TX, gt) or empty_after(WER_TX, pred):
                continue

            wer = jiwer.wer(gt, pred, truth_transform=WER_TX, hypothesis_transform=WER_TX)
            if wer >= 0.05:
                continue

            cer = jiwer.cer(gt, pred, truth_transform=CER_TX, hypothesis_transform=CER_TX)
            path_image = guess_image_path(path_audio)

            items.append({
                "idx": i,
                "audio": os.path.join(ROOT_PATH, path_audio),
                "image": os.path.join(ROOT_PATH, path_image),
                "gt": gt,
                "wer": wer,
                "cer": cer
            })

    if not items:
        print("Không có mẫu nào với WER < 5%.")
        return

    # Sắp xếp tăng dần theo WER
    items.sort(key=lambda x: x["wer"])

    not_found = []

    with open(name_text_path, "w", encoding="utf-8") as nt:
        for rank, it in enumerate(items, 1):
            audio_src = it["audio"]
            image_src = it["image"]
            name_file = f"{rank:04d}"
            audio_dst = os.path.join(DEST_DIR, f"{name_file}.wav")
            image_dst = os.path.join(DEST_DIR, f"{name_file}.png")

            # copy audio
            if os.path.isfile(audio_src):
                shutil.copy2(audio_src, audio_dst)
            else:
                not_found.append(("audio", audio_src))
                continue  # nếu thiếu audio thì bỏ

            # copy image
            if os.path.isfile(image_src):
                shutil.copy2(image_src, image_dst)
            else:
                not_found.append(("image", image_src))

            nt.write(f"{name_file}|{it['gt']}\n")

    print(f"Đã copy {len(items)} mẫu WER < 5% sang: {DEST_DIR}")
    print(f"File name|text: {name_text_path}")
    if not_found:
        print("⚠️  Một số file không tìm thấy:")
        for kind, p in not_found:
            print(f"  - {kind}: {p}")

if __name__ == "__main__":
    main()
