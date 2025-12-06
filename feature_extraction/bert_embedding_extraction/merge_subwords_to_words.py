import numpy as np

def merge_subwords_to_words(sentence, tokens, embeddings, offsets):
    """
    Convert BERT subword embeddings into one embedding per WORD.
    """

    words = sentence.split()
    word_embeddings = []
    current_subword_embs = []

    for tok, (start, end), emb in zip(tokens, offsets, embeddings):

        # Skip special tokens
        if tok in ["[CLS]", "[SEP]"]:
            continue

        # For a new word: if token is NOT a continuation ("##") AND we have collected previous subwords → finish them
        if not tok.startswith("##") and current_subword_embs:
            avg = np.mean(np.stack(current_subword_embs), axis=0)
            word_embeddings.append(avg)
            current_subword_embs = []

        # accumulate embeddings
        current_subword_embs.append(emb.cpu().numpy())

    # last word
    if current_subword_embs:
        avg = np.mean(np.stack(current_subword_embs), axis=0)
        word_embeddings.append(avg)

    return words, word_embeddings
