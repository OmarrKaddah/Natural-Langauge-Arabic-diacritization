def flatten_dataset(chars_sents, labels_sents, words_sents):
    """Input: lists of sentences (lists) of chars, labels, words
    Output: list of rows with sent_idx, global_index, word, char, label, pos_in_word, word_length
    Output example:[
  {
    "sent_idx": 0,
    "global_index": 0,
    "word": "كتاب",
    "char": "ك",
    "label": 3,
    "pos_in_word": 0,
    "word_length": 4
  },
  {
    "sent_idx": 0,
    "global_index": 1,
    "word": "كتاب",
    "char": "ت",
    "label": 1,
    "pos_in_word": 1,
    "word_length": 4
  },
    """
    rows = []
    global_idx = 0

    for sent_idx, (chars, labels, words) in enumerate(zip(chars_sents, labels_sents, words_sents)):

        # Reconstruct sentence characters by joining words
        # But we need correct alignment → assume chars is correct
        char_pointer = 0

        for w in words:
            w_chars = list(w)
            for pos_in_word, ch in enumerate(w_chars):
                rows.append({
                    "sent_idx": sent_idx,
                    "global_index": global_idx,
                    "word": w,
                    "char": chars[char_pointer],
                    "label": labels[char_pointer],
                    "pos_in_word": pos_in_word,
                    "word_length": len(w),
                })
                char_pointer += 1
                global_idx += 1

    return rows