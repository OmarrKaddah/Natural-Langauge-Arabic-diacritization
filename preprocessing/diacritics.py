import re

# Arabic diacritics unicode block



DIACRITICS_PATTERN = r"[\u0617-\u061A\u064B-\u0652]"
AR_DIACRITICS = re.compile(DIACRITICS_PATTERN)

def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics from text."""
    return AR_DIACRITICS.sub("", text)

def get_diacritic(char: str) -> str:
    """Return the diacritic(s) attached to a character, as a string."""
    return "".join(re.findall(DIACRITICS_PATTERN, char))
