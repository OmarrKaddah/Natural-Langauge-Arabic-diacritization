import json

def load_train_sentences(path="train_processed.json"):
    """
    Loads the dataset and extracts undiacritized sentences.
    Returns:
        sentences : list of strings
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "undiac" in item and isinstance(item["undiac"], str):
            sentences.append(item["undiac"].strip())

    return sentences
