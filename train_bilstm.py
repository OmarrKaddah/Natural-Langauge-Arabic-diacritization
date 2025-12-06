import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from data_loader import get_dataloaders
from bilstm_model import BiLSTMTagger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for x, y, lengths in tqdm(loader, desc="Training"):
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)

        optimizer.zero_grad()

        logits = model(x, lengths)  # (batch, seq_len, num_tags)

        # reshape for CE loss
        logits = logits.reshape(-1, logits.size(-1))
        y = y.view(-1)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y, lengths in tqdm(loader, desc="Evaluating"):
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)

            logits = model(x, lengths)

            logits = logits.reshape(-1, logits.size(-1))
            y = y.view(-1)

            loss = criterion(logits, y)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    train_loader, val_loader, vocab = get_dataloaders(batch_size=32)

    model = BiLSTMTagger(
        vocab_size=len(vocab.letter2id),
        embedding_dim=256,
        hidden_dim=256,
        num_tags=len(vocab.diacritic2id),
        pad_idx=vocab.letter2id["<PAD>"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        print(f"\nEpoch {epoch+1}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = eval_epoch(model, val_loader, criterion)

        print(f"Train loss={train_loss:.4f} | Val loss={val_loss:.4f}")

    torch.save(model.state_dict(), "bilstm_model.pt")
    print("Model saved.")

if __name__ == "__main__":
    main()
