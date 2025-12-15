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

        # -------------------
        # Morphological mappings
        # -------------------
        self.pos2id = {"UNK": 0, "NOUN": 1, "VERB": 2, "ADJ": 3, "PRON": 4, "ADV": 5} 
        self.gender2id = {"UNK": 0, "M": 1, "F": 2, "N": 3}
        self.number2id = {"UNK": 0, "SING": 1, "PLUR": 2, "DUAL": 3}
        self.aspect2id = {"UNK": 0, "PERF": 1, "IMPF": 2}

    def encode_chars(self, chars):
        return [self.char2id.get(ch, self.char2id["<UNK>"]) for ch in chars]

    def encode_diacritics(self, diacritics):
        return [self.diacritic2id.get(d, self.diacritic2id.get("", 0)) for d in diacritics]

    def decode_chars(self, ids):
        return [self.id2char[i] for i in ids]

    def decode_diacritics(self, ids):
        return [self.id2diacritic[i] for i in ids]

    def encode_pos(self, pos_seq):
        return [self.pos2id.get(p, 0) for p in pos_seq]

    def encode_gender(self, gender_seq):
        return [self.gender2id.get(g, 0) for g in gender_seq]

    def encode_number(self, number_seq):
        return [self.number2id.get(n, 0) for n in number_seq]

    def encode_aspect(self, aspect_seq):
        return [self.aspect2id.get(a, 0) for a in aspect_seq]
