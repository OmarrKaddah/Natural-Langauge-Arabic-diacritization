import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, sentences, labels, vocab, max_len=200):
        """
        sentences: list of list of characters
        labels: list of list of diacritics (same length as sentences)
        vocab: StandardVocab instance
        """
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

        self.PAD_CHAR = vocab.char2id["<PAD>"]
        self.PAD_LABEL = vocab.diacritic2id[""]

    def __len__(self):
        return len(self.sentences)

    def _pad(self, seq, pad_value):
        if len(seq) >= self.max_len:
            return seq[:self.max_len]
        return seq + [pad_value] * (self.max_len - len(seq))

    def __getitem__(self, idx):
        chars = self.sentences[idx]
        diacs = self.labels[idx]

        # Encode
        char_ids = self.vocab.encode_chars(chars)
        label_ids = self.vocab.encode_diacritics(diacs)

        # Pad
        char_ids = self._pad(char_ids, self.PAD_CHAR)
        label_ids = self._pad(label_ids, self.PAD_LABEL)

        # Mask
        mask = [1 if x != self.PAD_CHAR else 0 for x in char_ids]

        # Convert to tensors
        return (
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long)
        )
