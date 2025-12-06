import json
import torch
from torch.utils.data import Dataset

class DiacriticsDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        char_ids = torch.tensor(entry["char_ids"], dtype=torch.long)
        label_ids = torch.tensor(entry["label_ids"], dtype=torch.long)

        # Return undiacritized string for debugging
        undiac = entry["undiac"]

        return char_ids, label_ids, undiac
