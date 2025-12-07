from sklearn.feature_extraction import DictVectorizer
import joblib

def char_ngrams_from_word(word, pos, min_n=2, max_n=4):
    """ Extract character n-grams centered at position `pos` in `word`(building blocks making the word) 
    """
    
    feats = {}
    L = len(word)
    for n in range(min_n, max_n + 1):# from 2 to 4
        for i in range(max(0, pos-n+1), min(pos+1, L-n+1)):
            ng = word[i:i+n]
            feats[f"ng{n}_{ng}"] = 1
    return feats # return the n-grams as a dictionary


def fit_char_ngrams(rows, save_path):
    feat_dicts = []# list of feature dictionaries
    for r in rows:
        feat_dicts.append(char_ngrams_from_word(r["word"], r["pos_in_word"]))

    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(feat_dicts)# fit and transform the feature dictionaries into a sparse matrix
    joblib.dump(dv, save_path)
    return X, dv # return the sparse matrix and the vectorizer
