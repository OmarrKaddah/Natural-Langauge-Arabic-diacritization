# bilstm_crf.py

import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()

        # Character embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,  # ensures PAD embeddings don't get updated
        )

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # (batch, seq, feat)
        )

        # Linear layer maps BiLSTM output to tag scores (emissions)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

        # CRF layer
        self.crf = CRF(num_tags=tagset_size, batch_first=True)

    def _get_emissions(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        char_ids: (batch, seq_len)
        returns emissions: (batch, seq_len, num_tags)
        """
        embeds = self.embedding(char_ids)              # (B, T, E)
        lstm_out, _ = self.lstm(embeds)               # (B, T, 2H)
        emissions = self.fc(lstm_out)                 # (B, T, C)
        return emissions

    def forward(
        self,
        char_ids: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood loss.

        char_ids: (batch, seq_len)
        labels:   (batch, seq_len)
        mask:     (batch, seq_len) boolean or uint8

        returns: scalar loss
        """
        emissions = self._get_emissions(char_ids)
        # CRF returns log-likelihood; we want the negative
        log_likelihood = self.crf(emissions, labels, mask=mask, reduction="mean")
        return -log_likelihood

    def decode(
        self,
        char_ids: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Decode the best tag sequence using Viterbi.

        returns: list of list of tag indices, length per sequence
        """
        emissions = self._get_emissions(char_ids)
        pred_paths = self.crf.decode(emissions, mask=mask)
        return pred_paths
