import re

# Regex patterns
HTML_TAG = re.compile(r"<.*?>")
ENGLISH_LETTERS = re.compile(r"[A-Za-z]")
DIGITS = re.compile(r"[0-9٠-٩]")
PUNCTUATION = re.compile(r"[!@#$%^&*()_+=\[\]{};:\"\\|<>/?~،…»«]")
EXTRA_SPACES = re.compile(r"\s+")

def normalize_arabic(text: str) -> str:
    """
    Apply standard Arabic normalization.
    """
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    return text

def clean_text(text: str) -> str:
    """
    Clean Arabic text by removing HTML tags, English letters,
    digits, punctuation, and normalizing whitespace.
    """
    text = re.sub(HTML_TAG, " ", text)
    text = re.sub(ENGLISH_LETTERS, " ", text)
    text = re.sub(DIGITS, " ", text)
    text = re.sub(PUNCTUATION, " ", text)
    
    # Normalize Arabic
    text = normalize_arabic(text)

    # Collapse spaces
    text = re.sub(EXTRA_SPACES, " ", text).strip()

    return text
