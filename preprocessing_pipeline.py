import json
from tqdm import tqdm

from dataset import vocab
from preprocessing.preprocess_sentence import preprocess_sentence
from dataset.vocab import StandardVocab
from preprocessing.morph_features import extract_word_morphology
from preprocessing.align_features import align_word_features_to_chars


def process_file(input_path, output_path, vocab):
    dataset = []

    print(f"Processing {input_path} ...")

    with open(input_path, "r", encoding="utf8") as f:
        print("Extracting chars and labels ...")
        for line in tqdm(f):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # chars = list of characters, labels = list of diacritic strings
            
            chars, labels = preprocess_sentence(line)
            words, word_feats = extract_word_morphology("".join(chars))

            pos_seq, gender_seq, number_seq, aspect_seq = align_word_features_to_chars(
            chars, words, word_feats
            )
           

            # Encode
            char_ids = vocab.encode_chars(chars)
            label_ids = vocab.encode_diacritics(labels)
            pos_ids = vocab.encode_pos(pos_seq)
            gender_ids = vocab.encode_gender(gender_seq)
            number_ids = vocab.encode_number(number_seq)
            aspect_ids = vocab.encode_aspect(aspect_seq)
            entry = {
                "undiac": "".join(chars),
                "chars": chars,
                "labels": labels,
                "char_ids": char_ids,
                "label_ids": label_ids,
                "pos_ids": pos_ids,
                "gender_ids": gender_ids,
                "number_ids": number_ids,
                "aspect_ids": aspect_ids   
            }

            dataset.append(entry)
        print(f"Total sentences processed: {len(dataset)}")

    # Save JSON
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved → {output_path}")


def main():
    vocab = StandardVocab(base_path="dataset")

    process_file("dataset/train.txt", "train_processed.json", vocab)
    process_file("dataset/val.txt", "val_processed.json", vocab)


if __name__ == "__main__":
    main()
