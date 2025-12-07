import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim=128, hidden_dim=256):
        super().__init__()

        # 1. Character embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0   # PAD = 0
        )

        # 2. BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,   # because bidirectional
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 3. Linear layer (maps LSTM output → diacritic classes)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, mask):
        """
        input_ids: (batch, seq_len)
        mask:      (batch, seq_len) 1 = real token, 0 = PAD
        """

        # Step A: Embeddings
        emb = self.embedding(input_ids)        # (B, L, embed_dim)

        # Step B: LSTM
        lstm_out, _ = self.lstm(emb)           # (B, L, hidden_dim)

        # Step C: Linear layer
        logits = self.classifier(lstm_out)     # (B, L, num_labels)

        # We will handle softmax + loss outside for now
        return logits
