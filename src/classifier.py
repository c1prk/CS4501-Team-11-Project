"""
classifier.py — Improved Bag of Visual Words + Spatial Pyramid

Upgrades over the original classifier.py, in order of impact:

  1. Vocabulary size k=500 instead of 200 (matches the slides, and the extra
     granularity helps separate GAN textures from real skin).
  2. Spatial pyramid: split the face into a 2x2 grid of regions (eyes-left,
     eyes-right, mouth-left, mouth-right) and compute a histogram per region.
     Concatenated with the global histogram -> 5 * k features. This captures
     WHERE unusual textures occur, not just which textures exist. GAN blending
     artifacts tend to cluster around specific facial regions.
  3. Hand-crafted features concatenated: Canny edge density (4-d) and HSV
     color stats (6-d) appended to the histogram. This was in the midterm's
     "Planned Improvements" list but never actually wired up.
  4. Larger training-sample pool for vocabulary (5000 images, not 2000).
  5. L2 normalization instead of L1 — plays better with RBF-SVM.

Expected: mid-60s -> low-70s. Classical single-frame caps around there.

Usage (from project root):
  python -m src.classifier
"""
import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# --- paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "data", "weights")

# --- config ---
VOCAB_SIZE = 500               # bumped from 200
VOCAB_SAMPLE = 5000            # bumped from 2000
MAX_DESCRIPTORS_PER_IMG = 150  # bumped from 100
PYRAMID_GRID = (2, 2)          # 2x2 spatial pyramid on top of the global hist

# --- derived dims ---
N_REGIONS = 1 + PYRAMID_GRID[0] * PYRAMID_GRID[1]  # 1 global + 4 quadrants = 5
EXTRA_FEATURES = 10  # 4 edge + 6 HSV


# ======================================================================
# file indexing + split loading (same pattern as original)
# ======================================================================

def build_file_index():
    index = {}
    for root, _, files in os.walk(PROCESSED_DIR):
        for f in files:
            if f.endswith('.jpg'):
                full_path = os.path.join(root, f)
                label = 1 if 'manipulated_frames' in full_path else 0
                index.setdefault(f, []).append((full_path, label))
    return index


def load_split(split_name, index):
    split_file = os.path.join(SPLITS_DIR, f"{split_name}.txt")
    with open(split_file) as f:
        filenames = [line.strip() for line in f if line.strip()]
    samples, seen = [], {}
    for fname in filenames:
        if fname in index:
            c = seen.get(fname, 0)
            if c < len(index[fname]):
                samples.append(index[fname][c])
                seen[fname] = c + 1
    labels = [s[1] for s in samples]
    print(f"  {split_name}: {len(samples)} images "
          f"({sum(labels)} fake, {len(labels)-sum(labels)} real)")
    return samples


# ======================================================================
# SIFT extraction — returns keypoints AND descriptors so we can bin by
# region for the spatial pyramid.
# ======================================================================

def extract_sift_kps_and_descs(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return img, keypoints, descriptors


# ======================================================================
# Vocabulary building
# ======================================================================

def build_vocabulary(samples):
    """Collect descriptors from training images, cluster into visual words."""
    print(f"\n[Vocab] building from up to {VOCAB_SAMPLE} training images...")
    rng = np.random.RandomState(42)

    if len(samples) > VOCAB_SAMPLE:
        idx = rng.choice(len(samples), VOCAB_SAMPLE, replace=False)
        samples = [samples[i] for i in idx]

    all_descriptors = []
    for i, (path, _) in enumerate(samples):
        _, _, des = extract_sift_kps_and_descs(path)
        if des is not None and len(des) > 0:
            if len(des) > MAX_DESCRIPTORS_PER_IMG:
                sel = rng.choice(len(des), MAX_DESCRIPTORS_PER_IMG, replace=False)
                des = des[sel]
            all_descriptors.append(des)
        if (i + 1) % 500 == 0:
            print(f"  descriptors from {i+1}/{len(samples)}")

    all_descriptors = np.vstack(all_descriptors).astype(np.float32)
    print(f"  total descriptors: {len(all_descriptors)}")

    # K-Means caps: 100k is plenty for k=500 clusters
    if len(all_descriptors) > 100_000:
        sel = rng.choice(len(all_descriptors), 100_000, replace=False)
        all_descriptors = all_descriptors[sel]

    print(f"  K-Means k={VOCAB_SIZE}...")
    kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, random_state=42,
                             batch_size=2048, n_init=5)
    kmeans.fit(all_descriptors)
    print("  vocabulary built")
    return kmeans


# ======================================================================
# Spatial pyramid histogram
# ======================================================================

def _region_index(x, y, w, h, grid):
    """Return which cell (row, col) of a grid a point (x,y) falls into."""
    gr, gc = grid
    r = min(int(y / h * gr), gr - 1)
    c = min(int(x / w * gc), gc - 1)
    return r * gc + c


def spatial_pyramid_histogram(image_path, kmeans):
    """
    Returns a concatenated histogram:
      [global_hist_k, region_0_hist_k, region_1_hist_k, ..., region_N-1_hist_k]
    Length = (1 + gr*gc) * VOCAB_SIZE
    """
    img, kps, des = extract_sift_kps_and_descs(image_path)
    dim = N_REGIONS * VOCAB_SIZE
    hists = np.zeros((N_REGIONS, VOCAB_SIZE), dtype=np.float32)

    if img is None or des is None or len(des) == 0:
        return hists.flatten()

    h, w = img.shape
    words = kmeans.predict(des.astype(np.float32))
    gr, gc = PYRAMID_GRID

    for kp, word in zip(kps, words):
        # region 0 = global (every keypoint goes here)
        hists[0, word] += 1
        x, y = kp.pt
        region_id = 1 + _region_index(x, y, w, h, (gr, gc))
        hists[region_id, word] += 1

    # L2 normalize each region's histogram separately, then flatten
    for i in range(N_REGIONS):
        norm = np.linalg.norm(hists[i])
        if norm > 0:
            hists[i] /= norm

    return hists.flatten()


# ======================================================================
# Handcrafted features (edge density + HSV stats)
# ======================================================================

def handcrafted_features(image_path):
    """Edge density stats (4) + HSV channel means/stds (6) = 10 features."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(EXTRA_FEATURES, dtype=np.float32)

    # edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    h, w = edges.shape
    overall = np.count_nonzero(edges) / (h * w)
    center = edges[h//4:3*h//4, w//4:3*w//4]
    center_d = np.count_nonzero(center) / max(1, center.size)
    border_pix = h*w - center.size
    border_d = (np.count_nonzero(edges) - np.count_nonzero(center)) / max(1, border_pix)
    boundary_ratio = border_d / (center_d + 1e-6)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    return np.array([
        overall, center_d, border_d, boundary_ratio,
        hh.mean(), hh.std(), ss.mean(), ss.std(), vv.mean(), vv.std()
    ], dtype=np.float32)


# ======================================================================
# Full feature vector per image
# ======================================================================

def encode_image(image_path, kmeans):
    """Concatenate spatial pyramid BoVW with handcrafted features."""
    pyramid = spatial_pyramid_histogram(image_path, kmeans)
    extras = handcrafted_features(image_path)
    return np.concatenate([pyramid, extras])


def encode_split(samples, kmeans, name=""):
    X, y = [], []
    for i, (path, label) in enumerate(samples):
        X.append(encode_image(path, kmeans))
        y.append(label)
        if (i + 1) % 1000 == 0:
            print(f"  {name}: encoded {i+1}/{len(samples)}")
    return np.array(X, dtype=np.float32), np.array(y)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("=== Improved BoVW + Spatial Pyramid + Handcrafted ===\n")
    print(f"Config: k={VOCAB_SIZE}  regions={N_REGIONS}  "
          f"final dim = {N_REGIONS*VOCAB_SIZE + EXTRA_FEATURES}\n")

    print("Indexing images...")
    index = build_file_index()
    print(f"  {len(index)} filenames\n")

    train_samples = load_split("train", index)
    val_samples   = load_split("val", index)
    test_samples  = load_split("test", index)

    kmeans = build_vocabulary(train_samples)

    print("\nEncoding images...")
    X_train, y_train = encode_split(train_samples, kmeans, "train")
    X_val,   y_val   = encode_split(val_samples, kmeans, "val")
    X_test,  y_test  = encode_split(test_samples, kmeans, "test")
    print(f"  feature dim: {X_train.shape[1]}")

    # normalize (fit on train only — prevent leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # --- Mini grid search on (C, gamma) using val set ---
    print("\nGrid-searching SVM hyperparameters on validation set...")
    best = {'acc': -1}
    for C in [0.5, 1.0, 2.0, 4.0]:
        for gamma in ['scale', 0.01, 0.001]:
            svm = SVC(kernel='rbf', C=C, gamma=gamma,
                      class_weight='balanced', random_state=42)
            svm.fit(X_train, y_train)
            val_acc = accuracy_score(y_val, svm.predict(X_val))
            print(f"  C={C}  gamma={gamma}  val_acc={val_acc:.4f}")
            if val_acc > best['acc']:
                best = {'acc': val_acc, 'C': C, 'gamma': gamma, 'svm': svm}

    print(f"\nBest: C={best['C']}  gamma={best['gamma']}  val_acc={best['acc']:.4f}")
    svm = best['svm']

    # --- Refit with probability=True for AUC on best hyperparams ---
    svm = SVC(kernel='rbf', C=best['C'], gamma=best['gamma'],
              class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    print("\n--- Test ---")
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"AUC:      {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake'], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # save artifacts
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    joblib.dump(kmeans, os.path.join(WEIGHTS_DIR, "bovw_vocab.pkl"))
    joblib.dump(svm,    os.path.join(WEIGHTS_DIR, "bovw_svm.pkl"))
    joblib.dump(scaler, os.path.join(WEIGHTS_DIR, "bovw_scaler.pkl"))
    print(f"\nSaved: bovw_vocab.pkl, bovw_svm.pkl, bovw_scaler.pkl in {WEIGHTS_DIR}/")