# embedding_extraction/build_features.py

from .bert_extractor import BertFeatureExtractor
from .aligner import align_subwords_to_characters

def build_features(undiac_sentence, chars):
    """
    Input:
        undiac_sentence: string (from preprocessing)
        chars: list of characters (from preprocessing)

    Output:
        embeddings: list of 768-d vectors aligned to chars
    """

    bert = BertFeatureExtractor()

    bert_tokens, bert_embs = bert.encode(undiac_sentence)

    embeddings = align_subwords_to_characters(chars, bert_tokens, bert_embs)

    return embeddings
