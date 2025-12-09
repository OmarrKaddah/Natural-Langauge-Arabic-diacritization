import torch
from models.bilstm_crf import BiLSTM_CRF
from dataset.vocab import StandardVocab


def load_model(model_path="bilstm_crf_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = StandardVocab()

    model = BiLSTM_CRF(
        vocab_size=len(vocab.letters),
        num_labels=len(vocab.diacritic2id),
        embed_dim=128,
        hidden_dim=256
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, vocab, device


def diacritize(model, vocab, device, text, window=60, stride=50):
    """
    Diacritizes LONG Arabic text using sliding windows.
    Matches the training RealDataset behavior.
    """

    chars = list(text)
    N = len(chars)

    # Encode full text
    encoded = vocab.encode_chars(chars)

    windows_outputs = []

    # Slide over text
    for start in range(0, N, stride):
        end = start + window
        window_chars = encoded[start:end]

        if len(window_chars) == 0:
            continue

        mask = [True] * len(window_chars)

        char_tensor = torch.tensor([window_chars], dtype=torch.long).to(device)
        mask_tensor = torch.tensor([mask], dtype=torch.bool).to(device)

        preds = model(char_tensor, mask_tensor)
        pred_ids = preds[0]

        diacs = [vocab.id2diacritic[d] for d in pred_ids]
        window_text = "".join([c + d for c, d in zip(chars[start:end], diacs)])

        windows_outputs.append(window_text)

    # Merge windows (remove overlaps)
    if len(windows_outputs) == 0:
        return ""

    final_output = windows_outputs[0]

    for i in range(1, len(windows_outputs)):
        overlap = window - stride  # typically 10 chars
        final_output += windows_outputs[i][overlap * 2:]  # keep new part only

    return final_output


def write_output_to_file(input_text, output_text, file_path="output.txt"):
    with open(file_path, "w", encoding="utf8") as f:
        f.write("Input:\n")
        f.write(input_text + "\n\n")
        f.write("Diacritized Output:\n")
        f.write(output_text + "\n")
    print(f"Saved output to {file_path}")


if __name__ == "__main__":
    model, vocab, device = load_model()

    example = (
        "الشهادة ظاهرة وبحق بين تضعف التهمة وهو الفرق بينه وبين الشهادة "
        "وعن أصبغ الجواز في الولد والزوجة والأخ والمكاتب والمدبر والمديان "
        "إن كان من أهل القيام بالحق وصح الحكم وقد يحكم للخليفة وهو فوقه "
        "وتهمته أقوى ولا ينبغي له القضاء بين أحد من عشيرته"
    )

    print("Input:", example)
    
    output = diacritize(model, vocab, device, example, window=60, stride=50)
    
    print("Output:", output)

    write_output_to_file(example, output, "diacritized_output.txt")
