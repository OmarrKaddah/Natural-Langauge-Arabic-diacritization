import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim=128, hidden_dim=256):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )

        # Linear layer → emission scores
        self.fc = nn.Linear(hidden_dim, num_labels)

        # CRF layer
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, mask, labels=None):
        # Embedding
        emb = self.embedding(input_ids)   # (B, L, E)

        # BiLSTM
        lstm_out, _ = self.lstm(emb)      # (B, L, H)

        # Linear → logits
        emissions = self.fc(lstm_out)     # (B, L, num_labels)

        if labels is not None:
            # Training: return loss (negative log-likelihood)
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss

        else:
            # Inference: decode best sequence
            return self.crf.decode(emissions, mask=mask)
