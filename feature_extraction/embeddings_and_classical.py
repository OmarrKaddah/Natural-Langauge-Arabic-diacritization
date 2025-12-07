import json
import pickle
import numpy as np
from scipy.sparse import load_npz, hstack, csr_matrix, save_npz

# ==========================================
# 1. LOAD CLASSICAL FEATURES
# ==========================================
X_classical_path = "artifacts/X_train.npz"
y_path = "artifacts/y_train.npy"

print("Loading classical features...")
X_classical = load_npz(X_classical_path)  # sparse matrix
y = np.load(y_path)                       # labels
print("Classical features shape:", X_classical.shape)
print("Labels shape:", y.shape)

# ==========================================
# 2. LOAD EMBEDDINGS
# ==========================================
embedding_path = "combined_embeddings.pkl"

print("Loading embeddings...")
with open(embedding_path, "rb") as f:
    combined_data = pickle.load(f)

# Flatten embeddings per character
emb_list = []
for sent_data in combined_data:
    # Each word embedding is (1068,)
    for i, word in enumerate(sent_data["words"]):
        # Repeat embedding for each character in the word
        char_emb = np.tile(sent_data["embeddings"][i], (len(word), 1))
        emb_list.extend(char_emb)

X_embeddings = np.vstack(emb_list).astype(np.float32)
print("Embeddings shape:", X_embeddings.shape)

# ==========================================
# 3. ALIGNMENT CHECK
# ==========================================
if X_classical.shape[0] != X_embeddings.shape[0]:
    raise ValueError(
        f"Number of characters mismatch! "
        f"Classical: {X_classical.shape[0]}, Embeddings: {X_embeddings.shape[0]}"
    )
print("Alignment check passed.")

# ==========================================
# 4. COMBINE FEATURES
# ==========================================
print("Combining classical features with embeddings...")
X_combined = hstack([X_classical, csr_matrix(X_embeddings)], format="csr")
print("Combined feature matrix shape:", X_combined.shape)

# ==========================================
# 5. SAVE COMBINED FEATURES
# ==========================================
save_npz("artifacts/X_train_full.npz", X_combined)
np.save("artifacts/y_train_full.npy", y)
print("\n✔ Combined features saved:")
print("→ artifacts/X_train_full.npz")
print("→ artifacts/y_train_full.npy")
print("Ready for model training!")
