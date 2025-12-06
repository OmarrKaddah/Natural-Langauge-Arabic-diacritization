import torch
import torch.nn as nn

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x, lengths):
        # x: (batch, seq_len)

        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        # output: (batch, seq_len, 2*hidden_dim)

        logits = self.fc(output)  # (batch, seq_len, num_tags)

        return logits
