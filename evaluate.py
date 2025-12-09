import torch

def evaluate(model, dataloader, device, vocab=None, dump_path=None, max_batches=None):
    """
    Computes accuracy.
    If vocab and dump_path are provided → also creates a prediction dump.
    """

    model.eval()
    total_correct = 0
    total_count = 0

    # For dumping
    dump_lines = []
    do_dump = vocab is not None and dump_path is not None

    if do_dump:
        id2char = vocab.id2char
        id2diac = vocab.id2diacritic

    with torch.no_grad():
        for batch_idx, (char_ids, mask, label_ids, undiac_list) in enumerate(dataloader):

            char_ids = char_ids.to(device)
            mask      = mask.to(device)
            label_ids = label_ids.to(device)

            # CRF decode
            preds = model(char_ids, mask)   # list of lists

            B, T = char_ids.shape

            # Build pred tensor for accuracy
            pred_tensor = torch.zeros_like(label_ids)

            for i, seq in enumerate(preds):
                pred_tensor[i, :len(seq)] = torch.tensor(seq, device=device)

            # Accuracy
            correct = ((pred_tensor == label_ids) & mask).sum().item()
            total   = mask.sum().item()

            total_correct += correct
            total_count  += total

            # ------------------------------
            # DUMP LOGIC (optional)
            # ------------------------------
            if do_dump:
                for i in range(B):
                    valid = mask[i].bool()

                    input_ids = char_ids[i][valid].tolist()
                    gold_ids  = label_ids[i][valid].tolist()
                    pred_ids  = preds[i]

                    chars = [id2char[c] for c in input_ids]
                    gold_d = [id2diac[d] for d in gold_ids]
                    pred_d = [id2diac[d] for d in pred_ids]

                    gold_text = "".join([c+d for c, d in zip(chars, gold_d)])
                    pred_text = "".join([c+d for c, d in zip(chars, pred_d)])
                    undiac = undiac_list[i]

                    status = "CORRECT" if gold_text == pred_text else "WRONG"

                    dump_lines.append("===================================")
                    dump_lines.append(f"STATUS: {status}")
                    dump_lines.append(f"Undiacritized: {undiac}")
                    dump_lines.append(f"Gold:          {gold_text}")
                    dump_lines.append(f"Predicted:     {pred_text}")
                    dump_lines.append("")

                if max_batches and (batch_idx + 1) >= max_batches:
                    break

    # Save prediction dump
    if do_dump:
        with open(dump_path, "w", encoding="utf8") as f:
            f.write("\n".join(dump_lines))
        print(f"[evaluate] Prediction dump saved to {dump_path}")

    return total_correct / total_count
