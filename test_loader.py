from torch.utils.data import DataLoader
from dataset_loader import DiacriticsDataset

def main():
    dataset = DiacriticsDataset("train_processed.json")
    loader = DataLoader(dataset, batch_size=3, shuffle=True)

    for batch_idx, batch in enumerate(loader):
        char_ids, label_ids, undiac = batch

        print("BATCH", batch_idx)
        print("char_ids:", char_ids)
        print("label_ids:", label_ids)
        print("undiac:", undiac)

        if batch_idx == 1:  # only print first two batches
            break


if __name__ == "__main__":
    main()
