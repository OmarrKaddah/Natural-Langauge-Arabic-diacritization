from dataset.vocab import StandardVocab

from .cleaning import clean_text
from .extract_labels import extract_chars_and_labels

def preprocess_sentence(raw_sentence: str):
    """
    Clean → extract chars → extract diacritic labels → produce:
      undiacritized_sentence, chars, labels
    """

    cleaned = clean_text(raw_sentence)

    # FIX HERE (only two returned values)
    chars, labels = extract_chars_and_labels(cleaned)

    # Create undiacritized sentence
    undiac = "".join(chars)

    return chars, labels



sent = "الشَّهَادَةِ ظَاهِرَةً ، وَبِحَقٍّ بَيِّنٍ تَضْعُفُ التُّهْمَةُ ، وَهُوَ الْفَرْقُ بَيْنَهُ وَبَيْنَ الشَّهَادَةِ "
chars, labels = extract_chars_and_labels(sent)
# for c, l in zip(chars, labels):
#     print(f"'{c}': '{l}'")

vocab = StandardVocab(base_path="dataset")
char_ids = vocab.encode_chars(chars)
label_ids = vocab.encode_diacritics(labels)
for c, l, cid, lid in zip(chars, labels, char_ids, label_ids):
    print(f"'{c}': '{l}' -> {cid}, {lid}")








