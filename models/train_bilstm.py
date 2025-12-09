from dataset.real_dataset import RealDataset
import torch
from torch.utils.data import DataLoader

from dataset.vocab import StandardVocab
from dataset_loader import CharDataset
from models.bilstm_crf import BiLSTM_CRF
from evaluate import evaluate


from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    # Wrap dataloader with tqdm for a progress bar
    progress = tqdm(dataloader, desc="Training", leave=False)

    for char_ids, mask, label_ids,undiac in progress:
        char_ids = char_ids.to(device)
        mask = mask.to(device).bool()
        label_ids = label_ids.to(device)

        loss = model(char_ids, mask, label_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar text
        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)



def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vocab = StandardVocab()



    train_dataset = RealDataset("train_processed.json", window=60, stride=50)
    val_dataset   = RealDataset("val_processed.json", window=60, stride=50)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = BiLSTM_CRF(
        vocab_size=len(vocab.letters),
        num_labels=len(vocab.diacritic2id),
        embed_dim=128,
        hidden_dim=256
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    print("Training BiLSTM-CRF...\n")

    for epoch in tqdm(range(10), desc="Epochs"):
        loss = train_epoch(model, train_loader, optimizer, device)
        
        val_acc = evaluate(model,val_loader,device,vocab=vocab,dump_path=f"val_dump_epoch_{epoch+1}.txt",max_batches=10  )    # remove this to dump everything


        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")
    
    torch.save(model.state_dict(), "bilstm_crf_model.pth")


if __name__ == "__main__":
    main()
