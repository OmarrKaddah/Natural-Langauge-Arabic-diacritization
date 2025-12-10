import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_Tagger(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim=128, hidden_dim=256, pad_idx=0):
        super().__init__()

        self.pad_idx = pad_idx

        # Embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,  # half → bidirectional makes total hidden_dim
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Output classifier
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, char_ids, mask, label_ids=None):
        """
        char_ids : [B, T]
        mask     : [B, T] (bool)
        label_ids: [B, T] or None
        """

        emb = self.embedding(char_ids)           # [B, T, E]
        lstm_out, _ = self.lstm(emb)             # [B, T, H]
        logits = self.fc(lstm_out)               # [B, T, num_labels]

        if label_ids is None:
            # Inference mode → return argmax predictions
            preds = logits.argmax(dim=-1)
            return preds

        # --- Loss ---
        # Flatten everything
        B, T, C = logits.shape

        logits_flat = logits.view(B*T, C)
        labels_flat = label_ids.view(B*T)
        mask_flat = mask.view(B*T)

        # Compute cross-entropy over non-padded positions
        loss = F.cross_entropy(
            logits_flat[mask_flat],
            labels_flat[mask_flat]
        )

        return loss
