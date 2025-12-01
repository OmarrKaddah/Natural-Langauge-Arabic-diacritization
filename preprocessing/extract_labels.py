from .diacritics import get_diacritic, remove_diacritics

# Tag mapping for diacritics

# 0 = no diacritic
# 1 = fatha
# 2 = damma
# 3 = kasra
# 4 = sukun
# 5 = shadda
# 6 = tanween
# 7 = mad
# 8 = shadda+fatha
# 9 = shadda+damma
# 10 = shadda+kasra

DIACRITIC_TAGS = {
    "": 0,
    " َ": 1,
    " ُ": 2,
    " ِ": 3,
    " ْ": 4,
    " ّ": 5,
    "~": 7,
    " َّ": 8,
    " ُّ": 9,
    " ِّ": 10,
}



def extract_chars_and_labels(sentence: str):
    """
    Given a fully diacritized sentence:
      - returns list of base characters (no diacritics)
      - returns list of diacritic tag IDs
    """
    chars = []
    labels = []

    for ch in sentence:
        if ch.strip() == "":
            # Keep spaces as tokens? Yes for BiLSTM, but label = -1
            chars.append(" ")
            labels.append(-1)
            continue
        #FIX: returns emopty base for characters with diacritics
        dia = get_diacritic(ch)
        base = remove_diacritics(ch)
     
        
        print("dia=",dia)
        print("base=",base)
        if base == "":
            print("Skipping empty base character for diacritic:", dia)
            continue
        # print(dia,"test is=",DIACRITIC_TAGS.get(dia, 0))
        chars.append(base)
        labels.append(DIACRITIC_TAGS.get(dia, 0))
        print("diacritic=",dia)
        print("get returned=",DIACRITIC_TAGS.get(dia, 0))
        print("labels so far=",labels)
        

    return chars, labels
