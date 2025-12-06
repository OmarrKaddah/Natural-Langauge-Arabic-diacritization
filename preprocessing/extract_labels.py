# preprocessing/extract_labels.py

import re
import unicodedata
from .diacritics import DIACRITICS_PATTERN

import re
import unicodedata
from .diacritics import DIACRITICS_PATTERN

AR_DIACRITICS = re.compile(DIACRITICS_PATTERN)

# INSERT THIS:
DIACRITIC_ORDER = {
    "ّ": 0,
    "َ": 1, "ُ": 1, "ِ": 1, "ْ": 1,
    "ً": 2, "ٌ": 2, "ٍ": 2,
}

def sort_diacritics(d):
    return "".join(sorted(d, key=lambda x: DIACRITIC_ORDER.get(x, 99)))


AR_DIACRITICS = re.compile(DIACRITICS_PATTERN)


def is_diacritic(ch: str) -> bool:
    return bool(AR_DIACRITICS.fullmatch(ch))


def extract_chars_and_labels(sentence: str):
    """
    Returns:
        chars  -> list of base characters
        labels -> raw diacritic strings ( "", "َ", "ًّ", "ُّ", etc.)
    """

    chars = []
    labels = []

    i = 0
    n = len(sentence)

    while i < n:
        ch = sentence[i]

        # Space handling
        if ch.isspace():
            chars.append(" ")
            labels.append("")          # empty diacritic = class 14 in pickle
            i += 1
            continue

        # Skip stray diacritics
        if is_diacritic(ch):
            i += 1
            continue

        # Base character
        base = ch
        i += 1

        # Collect attached diacritics
        diacritics = ""
        while i < n and is_diacritic(sentence[i]):
            diacritics += sentence[i]
            i += 1

        # FIX: reorder diacritics into canonical order
        diacritics = sort_diacritics(diacritics)


        chars.append(base)
        labels.append(diacritics)

    return chars, labels
