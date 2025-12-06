# embedding_extraction/aligner.py

def align_subwords_to_characters(chars, bert_tokens, bert_embeddings):
    """
    Align BERT subword embeddings to original characters.

    chars: list like ['ب','ت','ح']
    bert_tokens: ['[CLS]', 'ب', '##ت', '##ح', '[SEP]']
    bert_embeddings: tensor of shape (seq_len, 768)

    Returns: list of embeddings, one per char
    """

    aligned_embeddings = []
    token_idx = 1  # skip [CLS]

    for ch in chars:
        # Use the next token (BERT subword)
        tok = bert_tokens[token_idx]

        # remove "##" prefix for matching
        if tok.startswith("##"):
            tok_clean = tok[2:]
        else:
            tok_clean = tok

        # Align character to its token embedding
        aligned_embeddings.append(bert_embeddings[token_idx].cpu())

        token_idx += 1

    return aligned_embeddings
