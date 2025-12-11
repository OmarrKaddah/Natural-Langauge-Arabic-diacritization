from feature_extraction.bert_embedding_extraction.bert_extractor import BertFeatureExtractor
from feature_extraction.bert_embedding_extraction.merge_subwords_to_words import merge_subwords_to_words

def build_word_features(sentence):
    """
    Takes ONE sentence → returns:
        words: list[str]
        word_embs: list[np.ndarray] (each 768 dims)
    """
    bert = BertFeatureExtractor()
    tokens, emb, offsets = bert.encode(sentence)
    words, word_embs = merge_subwords_to_words(sentence, tokens, emb, offsets)
    return words, word_embs


def build_features_for_dataset(sent_list):
    """
    Takes a LIST of sentences → returns list of dicts:
    [
       {"sentence": "...", "words": [...], "embeddings": [...]},
       ...
    ]
    """
    bert = BertFeatureExtractor()
    results = []

    for sent in sent_list:
        tokens, emb, offsets = bert.encode(sent)
        words, word_embs = merge_subwords_to_words(sent, tokens, emb, offsets)

        results.append({
            "sentence": sent,
            "words": words,
            "embeddings": word_embs
        })

    return results
