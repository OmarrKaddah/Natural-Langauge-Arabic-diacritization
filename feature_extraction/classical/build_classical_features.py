
from scipy.sparse import hstack, save_npz
import numpy as np
import joblib
from feature_extraction.classical.data_loader import load_dataset_for_classical
from feature_extraction.classical.flatten import flatten_dataset
from feature_extraction.classical.tfidf_word import fit_word_tfidf, transform_word_tfidf
from feature_extraction.classical.char_ngrams import fit_char_ngrams
from feature_extraction.classical.scalar_features import build_scalar_features
from sklearn.feature_selection import SelectKBest, chi2


def build_all_features(preprocessed_path, out_dir="artifacts/"):
    """
    Build all classical features directly from preprocessed JSON.
    
    Args:
        preprocessed_path: Path to preprocessed JSON file (train_processed.json)
        out_dir: Directory to save vectorizers and features
        train: If True, fit new vectorizers. If False, load existing ones.
    
    Returns:
        X: Feature matrix (sparse)
        y: Labels array
        rows: Flattened rows for reference
    """
    print("="*60)
    print("BUILDING CLASSICAL FEATURES")
    print("="*60)
    
    # Step 1: Load preprocessed data
    print("\n[1/7] Loading preprocessed dataset...")
    chars, labels, words = load_dataset_for_classical(preprocessed_path)
    print(f"     Loaded {len(chars)} sentences")

    # Step 2: Flatten to character-level
    print("\n[2/7] Flattening to character-level...")
    rows = flatten_dataset(chars, labels, words)
    print(f"      Created {len(rows)} character rows")

    # Step 3: Word TF-IDF features
    print("\n[3/7] Building word TF-IDF features...")
    
    
    tfidf = fit_word_tfidf(rows, out_dir + "vectorizer_tfidf.pkl")
    X_tfidf = transform_word_tfidf(rows, tfidf)
    print(f"      Fitted vectorizer: {X_tfidf.shape}")
    
    # Step 4: Character n-grams
    print("\n[4/7] Building character n-gram features...")
    
    X_ngrams, ngram_vec = fit_char_ngrams(rows, out_dir + "vectorizer_ngrams.pkl")
    print(f"      Fitted vectorizer: {X_ngrams.shape}")

    # Step 5: Scalar features
    print("\n[5/7] Building scalar features...")
    X_scalar = build_scalar_features(rows)
    print(f"      Created: {X_scalar.shape}")

    # Step 6: Combine all features
    print("\n[6/7] Combining all features...")
    X = hstack([X_tfidf, X_ngrams, X_scalar], format="csr")
    print(f"      Combined shape: {X.shape}")
    
    # Step 7: Feature selection
    print("\n[7/7] Applying feature selection...")
    #Apply SelectKBest Select top k features based on chi2
    SELECT_K = 3000
    selector_path = out_dir + "select_k_best.pkl"
    y = np.array([r["label"] for r in rows])
    selector = SelectKBest(score_func=chi2, k=SELECT_K)
    X_reduced = selector.fit_transform(X, y)
    joblib.dump(selector, selector_path)
    X = X_reduced
    print(f"      Reduced shape: {X.shape}")    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    print(f"Word TF-IDF:        {X_tfidf.shape[1]:>8,} features")
    print(f"Character n-grams:  {X_ngrams.shape[1]:>8,} features")
    print(f"Scalar features:    {X_scalar.shape[1]:>8,} features")
    print(f"{'─'*40}")
    print(f"TOTAL:              {X.shape[1]:>8,} features")
    print(f"Samples:            {X.shape[0]:>8,} characters")
    print(f"Sparsity:           {X.nnz / (X.shape[0] * X.shape[1]):>8.4%}")
    print("="*60)
    
    return X, y, rows


def save_features(X, y, out_dir="artifacts/", prefix="train"):
    """Save features and labels"""
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    save_npz(f"{out_dir}X_{prefix}.npz", X)
    np.save(f"{out_dir}y_{prefix}.npy", y)
    
    print(f"\n✓ Saved features to {out_dir}")
    print(f"  - X_{prefix}.npz")
    print(f"  - y_{prefix}.npy")


# Main execution functions
def build_train_features(preprocessed_path="train_processed.json", 
                        out_dir="artifacts/"):
    """Build and save training features"""
    
    print("\n🚀 BUILDING TRAINING FEATURES\n")
    
    X_train, y_train, rows = build_all_features(
        preprocessed_path=preprocessed_path,
        out_dir=out_dir,
        train=True  # Fit vectorizers
    )
    
    save_features(X_train, y_train, out_dir, prefix="train")
    
    return X_train, y_train, rows