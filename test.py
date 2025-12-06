import pickle
from preprocessing.extract_labels import extract_chars_and_labels
import unicodedata

letters = pickle.load(open("dataset/arabic_letters.pickle", "rb"))
diacritics = pickle.load(open("dataset/diacritics.pickle", "rb"))
diacritic2id = pickle.load(open("dataset/diacritic2id.pickle", "rb"))

sentences = [
    "قَوْمٌ",
    "ثُمَّ",
    "يُصَلُّونَ",
]

for s in sentences:
    print("\nSentence:", s)
    chars, labels = extract_chars_and_labels(s)
    

    def show_unicode(s):
        """Return each character with its unicode codepoint."""
        return " ".join([f"{repr(ch)}(U+{ord(ch):04X})" for ch in s])


    for c, d in zip(chars, labels):

        print("\n---------------------------")
        print(f"Character: {c}")
        print(f"Extracted diacritic: '{d}'")
        print("Extracted codepoints:", show_unicode(d))

        found = False
        for key in diacritic2id.keys():
            if d == key:
                found = True
                print(f"  Key '{key}' → codepoints: {show_unicode(key)}")

        if d not in diacritic2id:
            print("❌ NOT IN diacritic2id")
        else:
            print(f"✔ ID = {diacritic2id[d]}")

