import pdb
import re
import random
import hashlib
from typing import List, Dict, Optional

EMOTION_CLASSES = [
    "ADORATION","AMAZEMENT","AMUSEMENT","ANGER","CONFUSION","CONTENTMENT",
    "CUTENESS","DESIRE","DISAPPOINTMENT","DISGUST","DISTRESS","EMBARASSMENT",
    "EXSTASY",
    "FEAR","GUILT","INTEREST","NEUTRAL","PAIN","PRIDE","REALIZATION","RELIEF",
    "SADNESS","SERENITY","FAST","SLOW","LOUD","WHISPER","LOWPITCH","HIGHPITCH"
]

N_PROMPTS = 6
SEED: Optional[int] = 42

NORMALIZE_MAP = {
    "EMBARRASSMENT": "EMBARASSMENT",
    "EMBARASSMENT": "EMBARASSMENT",
    "ECSTASY": "EXSTASY",
    "EXTASY": "EXSTASY",
    "EXSTASY": "EXSTASY",
    "FAST": "FAST",
    "SLOW": "SLOW",
    "LOUD": "LOUD",
    "WHISPER": "WHISPER",
    "WHISP": "WHISPER",
    "WHISPERING": "WHISPER",
    "LOWPITCH": "LOWPITCH",
    "LOW-PITCH": "LOWPITCH",
    "LOW_PITCH": "LOWPITCH",
    "LOW": "LOW",
    "HIGHPITCH": "HIGHPITCH",
    "HIGH-PITCH": "HIGHPITCH",
    "HIGH_PITCH": "HIGHPITCH",
    "HIGH": "HIGH",
    "PITCH": "PITCH",
}

TEMPLATES = [
    "Deliver this line in **{CLS}**.",
    "Keep the read centered on **{CLS}**.",
    "Maintain a consistent **{CLS}** tone.",
    "Speak with **{CLS}** throughout.",
    "Aim for a controlled **{CLS}** delivery.",
    "Read as **{CLS}** from start to finish.",
    "Focus the performance on **{CLS}**.",
    "Anchor the read in **{CLS}**.",
    "Project a clear **{CLS}** mood.",
    "Stay strictly **{CLS}** across the take.",
    "Keep pacing and emphasis in **{CLS}**.",
    "Consistently voice **{CLS}**."
]

def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

EMOTION_SET = set(_dedupe(EMOTION_CLASSES))

def _canonical(token: str) -> str:
    up = token.upper()
    can = NORMALIZE_MAP.get(up, up)
    if can in EMOTION_SET:
        return can
    if up in EMOTION_SET:
        return up
    return ""

def _extract_class_from_filename(filename: str) -> str:
    tokens = [t for t in re.split(r"[-_]+", filename) if t]
    U = [t.upper() for t in tokens]

    for i, tok in enumerate(U):
        if tok == "EMO" and i + 1 < len(U):
            cand = _canonical(U[i + 1])
            if cand:
                return cand
        m = re.fullmatch(r"EMO([A-Z]+)", tok)
        if m:
            cand = _canonical(m.group(1))
            if cand:
                return cand

    parts = set(U)

    if "LOW" in parts and "PITCH" in parts:
        cand = _canonical("LOWPITCH")
        if cand:
            return cand
    if "HIGH" in parts and "PITCH" in parts:
        cand = _canonical("HIGHPITCH")
        if cand:
            return cand

    for tok in U:
        cand = _canonical(tok)
        if cand in {"FAST", "SLOW", "LOUD", "WHISPER", "LOWPITCH", "HIGHPITCH"}:
            return cand

    return "NEUTRAL"

def _rng_from_seed(seed: Optional[int]) -> random.Random:
    if seed is None:
        return random.Random()
    return random.Random(seed)

def generate_prompts_for_class(emotion_cls: str, n: int = 6, seed: Optional[int] = None) -> List[str]:
    rng = random.Random()
    k = min(n, len(TEMPLATES))
    choices = rng.sample(TEMPLATES, k=k)
    return [t.format(CLS=emotion_cls) for t in choices]

def instruction_from_filename(filename: str, n_per_file=5) -> str:
    cls_ = _extract_class_from_filename(filename)
    out = generate_prompts_for_class(cls_, n=n_per_file)
    rng = random.Random()
    return rng.choice(out)

def instruction_from_keyword(cls_: str, n_per_file=5) -> str:
    out = generate_prompts_for_class(cls_, n=n_per_file)
    rng = random.Random()
    return rng.choice(out)

def keyword_from_filename(filename: str) -> str:
    return _extract_class_from_filename(filename)

if __name__ == "__main__":
    files = [
        "en-US-p088-emo_guilt_sentences",
        "en-US-p010-emo_desire_sentences",
        "en-US-p101-emo_anger_sentences",
        "en-US-p088-rainbow_06_fast",
        "en-US-p043-emo_contentment_sentences",
        "rainbow_01_highpitch",
        "rainbow_01_lowpitch",
        "sentences_04_whisper"
    ]
    for file in files:
        print(f"File: {file}")
        print("  Keyword:", keyword_from_filename(file))
        prompt = instruction_from_filename(file, n_per_file=6)
        print(" ", prompt)
