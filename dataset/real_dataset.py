import json
import torch
from torch.utils.data import Dataset

class RealDataset(Dataset):
    """
    Windowed dataset for Arabic diacritization.
    Splits each sentence into overlapping windows instead of truncating.
    """

    def __init__(self, json_path, window=60, stride=50):
        self.window = window
        self.stride = stride
        self.samples = []

        # Load JSON entries
        with open(json_path, "r", encoding="utf8") as f:
            data = json.load(f)

        # Build windowed samples
        for entry in data:
            char_ids = entry["char_ids"]
            label_ids = entry["label_ids"]
            undiac = entry["undiac"]

            L = len(char_ids)

            # If sentence short → one window only
            if L <= window:
                self.samples.append({
                    "char_ids": char_ids,
                    "label_ids": label_ids,
                    "undiac": undiac
                })
                continue

            # Otherwise: sliding windows
            for start in range(0, L, stride):
                end = start + window

                window_chars = char_ids[start:end]
                window_labels = label_ids[start:end]
                window_undiac = undiac[start:end]

                if len(window_chars) == 0:
                    continue

                self.samples.append({
                    "char_ids": window_chars,
                    "label_ids": window_labels,
                    "undiac": window_undiac
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        chars = item["char_ids"]
        labels = item["label_ids"]
        undiac = item["undiac"]

        seq_len = len(chars)
        pad_len = self.window - seq_len

        # Pad
        padded_chars = chars + [0] * pad_len
        padded_labels = labels + [14] * pad_len   # PAD label
        mask = [True] * seq_len + [False] * pad_len

        return (
            torch.tensor(padded_chars, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(padded_labels, dtype=torch.long),
            undiac   # keep original undecorated text slice
        )
