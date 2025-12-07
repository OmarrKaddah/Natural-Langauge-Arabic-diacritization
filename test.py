# from dataset_loader import CharDataset
# from dataset.vocab import StandardVocab

# vocab = StandardVocab()

# sentences = [
#     ['ذ','ه','ب','ا','ً']
# ]
# labels = [
#     ['َ','َ','َ','', 'ً']   # example labels
# ]

# dataset = CharDataset(sentences, labels, vocab, max_len=10)

# char_ids, mask, label_ids = dataset[0]

# print("char_ids: ", char_ids)
# print("mask:     ", mask)
# print("label_ids:", label_ids)

import torch
from models.bilstm import BiLSTMTagger
from dataset.vocab import StandardVocab
from dataset_loader import CharDataset

# 1. Setup vocab
vocab = StandardVocab()

# 2. Fake tiny dataset
sentences = [
    ['ذ','ه','ب','ا'],
]
labels = [
    ['َ','َ','َ',''],
]

dataset = CharDataset(sentences, labels, vocab, max_len=10)
char_ids, mask, label_ids = dataset[0]

# Add batch dimension
char_ids = char_ids.unsqueeze(0)  # (1, 10)
mask = mask.unsqueeze(0)          # (1, 10)

# 3. Build model
num_chars = len(vocab.letters)
num_labels = len(vocab.diacritic2id)
model = BiLSTMTagger(num_chars, num_labels)

# 4. Forward pass
logits = model(char_ids, mask)  # shape: (1, 10, num_labels)

print("logits shape:", logits.shape)
