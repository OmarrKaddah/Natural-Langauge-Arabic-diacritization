from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def fit_word_tfidf(rows, save_path):
    """ Fit a TF-IDF vectorizer on words from the flattened rows and save it """
    words = [r["word"] for r in rows]

    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=30000)
    tfidf.fit(words)
    joblib.dump(tfidf, save_path)
    return tfidf

def transform_word_tfidf(rows, tfidf):
    """Function to transform words using a fitted TF-IDF vectorizer
       Example: # Feature names
        ['السلام', 'عليكم', 'مرحبا']
        # Sparse matrix as array
[[0.707, 0.0, 0.0],   # "السلام"        
 [0.0, 1.0, 0.0],     # "عليكم"       
 [0.0, 0.0, 1.0],     # "مرحبا"       
 [0.707, 0.0, 0.0] again (lowered by TF-IDF normalization)  # "السلام"       """
 
    words = [r["word"] for r in rows]
    return tfidf.transform(words)
