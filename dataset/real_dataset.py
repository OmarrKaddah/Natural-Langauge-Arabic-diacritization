import json
import torch
from torch.utils.data import Dataset

class RealDataset(Dataset):
    def __init__(self, json_path, max_len=200):
        """
        json_path: path to train_processed.json or val_processed.json
        """
        with open(json_path, "r", encoding="utf8") as f:
            self.data = json.load(f)

        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        char_ids = entry["char_ids"]
        label_ids = entry["label_ids"]

        L = len(char_ids)
        max_len = self.max_len

        if L > max_len:
            char_ids = char_ids[:max_len]
            label_ids = label_ids[:max_len]
            L = max_len

        # Padding
        pad_len = max_len - L

        char_ids = char_ids + [0] * pad_len
        label_ids = label_ids + [14] * pad_len  # 14 = PAD label_id in your example

        mask = [True] * L + [False] * pad_len

        return (
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(label_ids, dtype=torch.long)
        )
