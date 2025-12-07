import torch



def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for char_ids, mask, label_ids in dataloader:
            char_ids = char_ids.to(device)
            mask = mask.to(device)
            label_ids = label_ids.to(device)

            # CRF decode → list of lists
            preds = model(char_ids, mask)

            # Convert list of lists into tensor
            pred_tensor = torch.zeros_like(label_ids)
            for i, seq in enumerate(preds):
                pred_tensor[i, :len(seq)] = torch.tensor(seq, device=device)

            # Count matches on non-pad tokens
            correct = ((pred_tensor == label_ids) & mask).sum().item()
            total = mask.sum().item()

            total_correct += correct
            total_count += total

    return total_correct / total_count
