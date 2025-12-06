# dataset/vocab.py

import pickle
import os

class StandardVocab:
    def __init__(self, base_path="dataset"):
        # Load pickles
        letters_set = pickle.load(open(os.path.join(base_path, "arabic_letters.pickle"), "rb"))
        diacritics_set = pickle.load(open(os.path.join(base_path, "diacritics.pickle"), "rb"))
        self.diacritic2id = pickle.load(open(os.path.join(base_path, "diacritic2id.pickle"), "rb"))

        # Convert sets to sorted lists
        self.letters = ["<PAD>", "<UNK>"] + sorted(list(letters_set))
        self.char2id = {ch: i for i, ch in enumerate(self.letters)}
        self.id2char = {i: ch for i, ch in enumerate(self.letters)}

        # Reverse diacritic mapping
        self.id2diacritic = {v: k for k, v in self.diacritic2id.items()}

    def encode_chars(self, chars):
        return [self.char2id.get(ch, self.char2id["<UNK>"]) for ch in chars]

    def encode_diacritics(self, diacritics):
        return [self.diacritic2id.get(d, self.diacritic2id[""]) for d in diacritics]

    def decode_chars(self, ids):
        return [self.id2char[i] for i in ids]

    def decode_diacritics(self, ids):
        return [self.id2diacritic[i] for i in ids]
