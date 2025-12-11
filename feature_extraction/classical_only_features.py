import os
import json
import numpy as np
from tqdm import tqdm
from scipy.sparse import save_npz

from feature_extraction.classical.build_classical_features import build_all_features


# ===========================
# CONFIG
# ===========================
PREPROCESSED_PATH = "train_processed.json"
OUT_DIR = "artifacts_classical_only/"
SELECT_K = 500   # <-- Reasonable reduced dimensionality


# ===========================
# MAIN FUNCTION
# ===========================
def build_classical_only_features(preprocessed_path=PREPROCESSED_PATH,
                                  out_dir=OUT_DIR,
                                  k_best=SELECT_K):

    # Ensure output folder exists
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(" CLASSICAL-ONLY FEATURE PIPELINE (TF-IDF + N-GRAMS + SCALARS)")
    print("=" * 70)

    # STEP 1 — Build all classical features
    print("\n[1] Building classical features...")
    X_classical, y, rows = build_all_features(
        preprocessed_path=preprocessed_path,
        out_dir=out_dir
    )

    print("\n[INFO] Raw classical shape:", X_classical.shape)

    # STEP 2 — Optional sanity checks
    print("\n[2] Verifying rows...")
    assert X_classical.shape[0] == len(rows), "ERROR: Mismatch between rows and features"

    # STEP 3 — Save as-is
    print("\n[3] Saving classical-only matrices...")

    save_npz(os.path.join(out_dir, "X_train_classical.npz"), X_classical)
    np.save(os.path.join(out_dir, "y_train_classical.npy"), y)

    # Save metadata so character → word alignment is preserved
    meta = [
        {
            "sentence_index": r.get("sentence_index", r.get("sent_idx")),
            "word_index": r.get("word_index", r.get("w_idx")),
            "char_index": r.get("char_index", r.get("c_idx")),
            "char": r.get("char"),
            "word": r.get("word"),
            "label": r.get("label")
        }
        for r in rows
    ]

    with open(os.path.join(out_dir, "rows_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(" DONE — Classical-only features saved successfully")
    print("=" * 70)
    print(f"  → {out_dir}X_train_classical.npz")
    print(f"  → {out_dir}y_train_classical.npy")
    print(f"  → {out_dir}rows_meta.json\n")


# ===========================
# ENTRY POINT
# ===========================
if __name__ == "__main__":
    build_classical_only_features()
