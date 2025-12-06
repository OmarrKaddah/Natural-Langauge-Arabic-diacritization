# dataset_loader.py

import json
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence


class DiacriticsDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        entry = self.data[idx]

        char_ids = entry["char_ids"]       # list[int]
        label_ids = entry["label_ids"]     # list[int]
        undiac = entry["undiac"]           # string (for debugging / later)

        return {
            "char_ids": char_ids,
            "label_ids": label_ids,
            "undiac": undiac,
        }


def create_collate_fn(char_pad_idx: int, label_pad_idx: int):
    """
    Returns a collate function that:
      - pads char & label sequences
      - creates a mask
      - converts to tensors

    Output:
      char_ids: (batch, seq_len) LongTensor
      label_ids: (batch, seq_len) LongTensor
      mask: (batch, seq_len) BoolTensor
      undiac: list[str] (original strings, unpadded)
    """
    def collate(batch):
        # Lists of sequences
        char_seqs = [torch.tensor(item["char_ids"], dtype=torch.long) for item in batch]
        label_seqs = [torch.tensor(item["label_ids"], dtype=torch.long) for item in batch]
        undiac = [item["undiac"] for item in batch]

        # Pad sequences to max length in batch
        padded_chars = pad_sequence(
            char_seqs,
            batch_first=True,
            padding_value=char_pad_idx,
        )  # (B, T)

        padded_labels = pad_sequence(
            label_seqs,
            batch_first=True,
            padding_value=label_pad_idx,
        )  # (B, T)

        # Mask: True where char is not PAD
        mask = (padded_chars != char_pad_idx)

        return padded_chars, padded_labels, mask, undiac

    return collate
