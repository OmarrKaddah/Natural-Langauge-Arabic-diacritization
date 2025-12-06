# train_bilstm_crf.py

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from dataset.vocab import StandardVocab
from dataset_loader import DiacriticsDataset, create_collate_fn
from models.bilstm_crf import BiLSTMCRF


def main():
    # ---------------------
    # 1) Device
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------
    # 2) Vocab and sizes
    # ---------------------
    vocab = StandardVocab(base_path="dataset")

    vocab_size = len(vocab.letters)              # chars vocab size
    tagset_size = len(vocab.diacritic2id)        # number of diacritic classes

    char_pad_idx = vocab.char2id["<PAD>"]
    # For labels, we can use the class for "no diacritic" as pad
    label_pad_idx = vocab.diacritic2id[""]

    print("vocab_size:", vocab_size)
    print("tagset_size:", tagset_size)
    print("char_pad_idx:", char_pad_idx)
    print("label_pad_idx:", label_pad_idx)

    # ---------------------
    # 3) Dataset + DataLoader
    # ---------------------
    train_dataset = DiacriticsDataset("train_processed.json")
    val_dataset = DiacriticsDataset("val_processed.json")

    batch_size = 32

    train_collate = create_collate_fn(char_pad_idx, label_pad_idx)
    val_collate = create_collate_fn(char_pad_idx, label_pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collate,
    )

    # ---------------------
    # 4) Model + Optimizer
    # ---------------------
    model = BiLSTMCRF(
        vocab_size=vocab_size,
        tagset_size=tagset_size,
        embedding_dim=128,
        hidden_dim=256,
        padding_idx=char_pad_idx,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5
    # ---------------------
    # 5) Training loop
    # ---------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (char_ids, label_ids, mask, undiac) in enumerate(
            tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
        ):
            char_ids = char_ids.to(device)
            label_ids = label_ids.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            loss = model(char_ids, label_ids, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_epoch_loss = total_loss / max(1, len(train_loader))
        print(f"\nEpoch {epoch} finished. Avg training loss: {avg_epoch_loss:.4f}")

        # -----------------
        # 6) Simple validation loop (loss only)
        # -----------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for char_ids, label_ids, mask, undiac in val_loader:
                char_ids = char_ids.to(device)
                label_ids = label_ids.to(device)
                mask = mask.to(device)

                loss = model(char_ids, label_ids, mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch} | Validation loss: {avg_val_loss:.4f}")
        print("-" * 60)

    # ---------------------
    # 7) Save model
    # ---------------------
    torch.save(model.state_dict(), "bilstm_crf_diacritizer.pt")
    print("Model saved to bilstm_crf_diacritizer.pt")


if __name__ == "__main__":
    main()
