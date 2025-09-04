import pdb
import re
import random
import hashlib
from typing import List, Dict, Optional

# ======= INPUT (bạn giữ nguyên list này đúng như bạn có, chỉ bỏ NEUTRAL trùng) =======
EMOTION_CLASSES = [
    "ADORATION","AMAZEMENT","AMUSEMENT","ANGER","CONFUSION","CONTENTMENT",
    "CUTENESS","DESIRE","DISAPPOINTMENT","DISGUST","DISTRESS","EMBARASSMENT",  # normalized
    "EXSTASY",  # normalized
    "FEAR","GUILT","INTEREST","NEUTRAL","PAIN","PRIDE","REALIZATION","RELIEF",
    "SADNESS","SERENITY","FAST","SLOW","LOUD","WHISPER","LOWPITCH","HIGHPITCH"
]

# ======= CONFIG =======
# Mặc định sinh 6 câu/filename
N_PROMPTS = 6
# Seed chung để reproducible (None = random mỗi lần)
SEED: Optional[int] = 42

# Chuẩn hoá token -> class trong EMOTION_CLASSES
NORMALIZE_MAP = {
    # chính tả hay biến thể
    "EMBARRASSMENT": "EMBARASSMENT",
    "EMBARASSMENT": "EMBARASSMENT",
    "ECSTASY": "EXSTASY",
    "EXTASY": "EXSTASY",
    "EXSTASY": "EXSTASY",

    # tốc độ/độ cao/thì thầm
    "FAST": "FAST",
    "SLOW": "SLOW",
    "LOUD": "LOUD",
    "WHISPER": "WHISPER",
    "WHISP": "WHISPER",
    "WHISPERING": "WHISPER",

    "LOWPITCH": "LOWPITCH",
    "LOW-PITCH": "LOWPITCH",
    "LOW_PITCH": "LOWPITCH",
    "LOW": "LOW",      # dùng cặp LOW + PITCH bên dưới

    "HIGHPITCH": "HIGHPITCH",
    "HIGH-PITCH": "HIGHPITCH",
    "HIGH_PITCH": "HIGHPITCH",
    "HIGH": "HIGH",    # dùng cặp HIGH + PITCH bên dưới

    "PITCH": "PITCH",  # để phát hiện cặp HIGH/LOW + PITCH
}

# Tập template: KHÔNG thêm class mới; chỉ chèn {CLS} đúng y nguyên.
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

# ======= CORE =======
def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# membership set (không giữ thứ tự)
EMOTION_SET = set(_dedupe(EMOTION_CLASSES))

def _canonical(token: str) -> str:
    up = token.upper()
    can = NORMALIZE_MAP.get(up, up)
    if can in EMOTION_SET:
        return can
    if up in EMOTION_SET:
        return up
    return ""  # không khớp

def _extract_class_from_filename(filename: str) -> str:
    tokens = [t for t in re.split(r"[-_]+", filename) if t]
    U = [t.upper() for t in tokens]

    # 1) Ưu tiên pattern emo + <class> (vd: emo_guilt_sentences)
    for i, tok in enumerate(U):
        if tok == "EMO" and i + 1 < len(U):
            cand = _canonical(U[i + 1])
            if cand:
                return cand
        # trường hợp "EMOANGER" ít gặp nhưng hỗ trợ
        m = re.fullmatch(r"EMO([A-Z]+)", tok)
        if m:
            cand = _canonical(m.group(1))
            if cand:
                return cand

    # 2) Tìm trực tiếp token thuộc nhóm tốc độ/giọng
    #    hoặc cặp HIGH+PITCH / LOW+PITCH
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

    # 3) Không tìm thấy → dùng NEUTRAL (có trong list)
    return "NEUTRAL"

def _rng_from_seed(seed: Optional[int]) -> random.Random:
    """Random riêng, không ảnh hưởng global random."""
    if seed is None:
        return random.Random()
    return random.Random(seed)

def _stable_int_from_text(text: str, salt: Optional[int]) -> int:
    """Tạo số nguyên ổn định từ text (+ salt) để reproducible per filename."""
    h = hashlib.sha256()
    if salt is not None:
        h.update(str(salt).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big", signed=False)

def generate_prompts_for_class(emotion_cls: str, n: int = 6, seed: Optional[int] = None) -> List[str]:
    rng = _rng_from_seed(seed)
    # chọn ngẫu nhiên n template KHÔNG lặp
    k = min(n, len(TEMPLATES))
    # dùng rng.sample thay vì random.sample để không động vào global state
    choices = rng.sample(TEMPLATES, k=k)
    return [t.format(CLS=emotion_cls) for t in choices]

def instruction_from_filename(filename: str, n_per_file=5) -> str:
    cls_ = _extract_class_from_filename(filename)
    # seed ổn định theo filename + SEED
    local_seed = _stable_int_from_text(filename, SEED)
    out = generate_prompts_for_class(cls_, n=n_per_file, seed=local_seed)
    # take 1 random prompt trong out (vẫn dùng RNG ổn định theo filename)
    rng = _rng_from_seed(local_seed ^ 0xA5A5A5A5)
    return rng.choice(out)

def instruction_from_keyword(cls_: str, n_per_file=5) -> str:
    local_seed = _stable_int_from_text(cls_, SEED)
    out = generate_prompts_for_class(cls_, n=n_per_file, seed=local_seed)
    # take 1 random prompt trong out (vẫn dùng RNG ổn định theo filename)
    rng = _rng_from_seed(local_seed ^ 0xA5A5A5A5)
    return rng.choice(out)

def keyword_from_filename(filename: str) -> str:
    return _extract_class_from_filename(filename)

# ======= DEMO =======
if __name__ == "__main__":
    files = [
        "en-US-p088-emo_guilt_sentences",
        "en-US-p010-emo_desire_sentences",
        "en-US-p101-emo_anger_sentences",
        "en-US-p088-rainbow_06_fast",
        "en-US-p043-emo_contentment_sentences",
        "rainbow_01_highpitch",
        "rainbow_01_lowpitch"
        "sentences_04_whisper"
    ]
    for file in files:
        print(f"File: {file}")
        print("  Keyword:", keyword_from_filename(file))
        prompt = instruction_from_filename(file, n_per_file=6)
        print(" ", prompt)
