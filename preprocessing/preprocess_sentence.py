from .cleaning import clean_text
from .extract_labels import extract_chars_and_labels

def preprocess_sentence(raw_sentence: str):
    """
    Clean → extract chars → extract diacritic labels → produce:
      undiacritized_sentence, char_list, label_list
    """
    cleaned = clean_text(raw_sentence)
    chars, labels = extract_chars_and_labels(cleaned)

    undiac = "".join(chars)

    return undiac, chars, labels


def main():
    #test
    raw_sentence = "<p>السَّلامُ عَلَيْكُمْ! كيف حالُكَ اليوم؟</p>"
    undiac, chars, labels = preprocess_sentence(raw_sentence)
    print("Undiacritized Sentence:", undiac)
    print("Characters:", chars)
    print("Labels:", labels)

if __name__ == "__main__":
    main()