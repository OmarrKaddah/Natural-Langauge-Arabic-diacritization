import json

def load_preprocessed_json(path):
    """
    Load preprocessed JSON file directly.
    
    Input format:
    [
        {
            "undiac": "ولو جمع",
            "chars": ["و", "ل", "و", " ", "ج", "م", "ع"],
            "labels": ["َ", "َ", "ْ", "", "َ", "َ", "َ"],
            "char_ids": [35, 31, 35, 1, 13, 32, 26],
            "label_ids": [0, 0, 6, 14, 0, 0, 0]
        },
        ...
    ]
    
    Returns:
        List of sentence dictionaries
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def extract_words_from_chars(chars):
    """
    Extract words from character list by splitting on spaces.
    
    Args:
        chars: List of characters including spaces ["و", "ل", "و", " ", "ج", "م", "ع"]
    
    Returns:
        List of words (no spaces): ["ولو", "جمع"]
    """
    words = []
    current_word = []
    
    for char in chars:
        if char == ' ':
            if current_word:
                words.append(''.join(current_word))
                current_word = []
        else:
            current_word.append(char)
    
    # Add last word if exists
    if current_word:
        words.append(''.join(current_word))
    
    return words

def preprocess_sentence(sentence_data):
    """
    Preprocess a single sentence from the JSON format.
    
    Args:
        sentence_data: Dict with 'chars', 'labels', etc.
    
    Returns:
        Tuple of (chars_no_space, labels_no_space, words)
    """
    # Remove spaces from chars and labels
    chars_clean = []
    labels_clean = []
    
    for char, label in zip(sentence_data['chars'], sentence_data['labels']):
        if char != ' ':
            chars_clean.append(char)
            labels_clean.append(label)
    
    # Extract words
    words = extract_words_from_chars(sentence_data['chars'])
    
    return chars_clean, labels_clean, words


def load_dataset_for_classical(json_path="preprocessed.json"):
    """
    Load preprocessed JSON and prepare for classical feature extraction.
        
    Args:
        json_path: Path to preprocessed JSON file
    
    Returns:
        chars_sents: List of character lists (no spaces)
        labels_sents: List of label lists (no spaces)
        words_sents: List of word lists
    """
    data = load_preprocessed_json(json_path)
    
    chars_sents = []
    labels_sents = []
    words_sents = []
    
    for sent in data:
        chars, labels, words = preprocess_sentence(sent)
        chars_sents.append(chars)
        labels_sents.append(labels)
        words_sents.append(words)
    
    return chars_sents, labels_sents, words_sents
