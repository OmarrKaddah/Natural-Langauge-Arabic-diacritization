# preprocessing/extract_labels.py

from .diacritics import DIACRITICS_PATTERN
import re

# ===== Diacritic tag mapping =====
# 0 = no diacritic
# 1 = fatha
# 2 = damma
# 3 = kasra
# 4 = sukun
# 5 = shadda
# 6 = tanween (any of ً ٌ ٍ )
# 7 = mad
# 8 = shadda+fatha
# 9 = shadda+damma
# 10 = shadda+kasra

FATHA = "\u064e"           # َ
DAMMA = "\u064f"           # ُ
KASRA = "\u0650"           # ِ
SUKUN = "\u0652"           # ْ
SHADDA = "\u0651"          # ّ
TANWEEN_FATHA = "\u064b"   # ً
TANWEEN_DAMMA = "\u064c"   # ٌ
TANWEEN_KASRA = "\u064d"   # ٍ
MADDA = "\u0653"           # ٓ

DIACRITIC_TAGS = {
    "": 0,
    FATHA: 1,
    DAMMA: 2,
    KASRA: 3,
    SUKUN: 4,
    SHADDA: 5,
    TANWEEN_FATHA: 6,
    TANWEEN_DAMMA: 6,
    TANWEEN_KASRA: 6,
    MADDA: 7,
    SHADDA + FATHA: 8,
    SHADDA + DAMMA: 9,
    SHADDA + KASRA: 10,
}

AR_DIACRITICS = re.compile(DIACRITICS_PATTERN)


def is_diacritic(ch: str) -> bool:
    return bool(AR_DIACRITICS.fullmatch(ch))


def extract_chars_and_labels(sentence: str):
    """
    Given a fully diacritized sentence:
      - returns list of base characters (no diacritics)
      - returns list of diacritic tag IDs

    We iterate over the string by index:
      - If it's a space → keep it with label -1
      - If it's a diacritic alone → skip (stray)
      - If it's a base char → gather following diacritics and map to a tag
    """
    chars = []
    labels = []

    i = 0
    n = len(sentence)

    while i < n:
        ch = sentence[i]

        # Preserve spaces (or any pure whitespace) with label -1
        if ch.isspace():
            chars.append(" ")
            labels.append(-1)
            i += 1
            continue

        # Stray diacritic (without a base char before it): skip
        if is_diacritic(ch):
            i += 1
            continue

        # Now ch is a base character
        base = ch
        i += 1

        # Collect all diacritics attached to this base
        diacritics = ""
        while i < n and is_diacritic(sentence[i]):
            diacritics += sentence[i]
            i += 1

        # Map diacritic sequence to tag id
        tag_id = DIACRITIC_TAGS.get(diacritics, 0)

        chars.append(base)
        labels.append(tag_id)

    return chars, labels
