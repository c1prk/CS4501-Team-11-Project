
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
SPLITS_DIR    = os.path.join(PROJECT_ROOT, "data", "splits")
WEIGHTS_DIR   = os.path.join(PROJECT_ROOT, "data", "weights")

# --- config (matches report Table 2) ---
VOCAB_SIZE             = 500   # visual words
VOCAB_SAMPLE           = 5000  # training images sampled for vocabulary
MAX_DESCRIPTORS_PER_IMG = 150


# ======================================================================
# File indexing + split loading
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
# SIFT extraction
# ======================================================================

def extract_sift_descriptors(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img, None)
    return descriptors


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
        des = extract_sift_descriptors(path)
        if des is not None and len(des) > 0:
            if len(des) > MAX_DESCRIPTORS_PER_IMG:
                sel = rng.choice(len(des), MAX_DESCRIPTORS_PER_IMG, replace=False)
                des = des[sel]
            all_descriptors.append(des)
        if (i + 1) % 500 == 0:
            print(f"  descriptors from {i+1}/{len(samples)}")

    all_descriptors = np.vstack(all_descriptors).astype(np.float32)
    print(f"  total descriptors: {len(all_descriptors)}")

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
# BoVW histogram (flat, 500-d)
# ======================================================================

def bovw_histogram(image_path, kmeans):
    """Represent the image as a normalized 500-bin BoVW histogram."""
    des = extract_sift_descriptors(image_path)
    hist = np.zeros(VOCAB_SIZE, dtype=np.float32)
    if des is not None and len(des) > 0:
        words = kmeans.predict(des.astype(np.float32))
        for w in words:
            hist[w] += 1
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
    return hist


# ======================================================================
# Handcrafted features — edge density (4) + HSV stats (6) = 10
# ======================================================================

def handcrafted_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(10, dtype=np.float32)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    h, w  = edges.shape
    overall      = np.count_nonzero(edges) / (h * w)
    center       = edges[h//4:3*h//4, w//4:3*w//4]
    center_d     = np.count_nonzero(center) / max(1, center.size)
    border_pix   = h * w - center.size
    border_d     = (np.count_nonzero(edges) - np.count_nonzero(center)) / max(1, border_pix)
    boundary_ratio = border_d / (center_d + 1e-6)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    return np.array([
        overall, center_d, border_d, boundary_ratio,
        hh.mean(), hh.std(), ss.mean(), ss.std(), vv.mean(), vv.std()
    ], dtype=np.float32)


# ======================================================================
# Full 510-d feature vector per image
# ======================================================================

def encode_image(image_path, kmeans):
    """Concatenate flat BoVW (500) + handcrafted (10) = 510 dimensions."""
    hist   = bovw_histogram(image_path, kmeans)
    extras = handcrafted_features(image_path)
    return np.concatenate([hist, extras])


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
    print("=== Bag of Visual Words + SVM ===\n")
    print(f"Config: k={VOCAB_SIZE}, feature dim=510 "
          f"(500 BoVW + 4 edge + 6 HSV)\n")

    print("Indexing images...")
    index = build_file_index()
    print(f"  {len(index)} filenames\n")

    train_samples = load_split("train", index)
    val_samples   = load_split("val",   index)
    test_samples  = load_split("test",  index)

    kmeans = build_vocabulary(train_samples)

    print("\nEncoding images...")
    X_train, y_train = encode_split(train_samples, kmeans, "train")
    X_val,   y_val   = encode_split(val_samples,   kmeans, "val")
    X_test,  y_test  = encode_split(test_samples,  kmeans, "test")
    print(f"  feature dim: {X_train.shape[1]}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Fixed hyperparameters matching Table 2 of the report
    print("\nTraining SVM (C=1.0, gamma=scale, class_weight=balanced)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale',
              class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    print("\n--- Validation ---")
    y_val_pred = svm.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred)*100:.2f}%")

    print("\n--- Test ---")
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"AUC:      {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=['Real', 'Fake'], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    joblib.dump(kmeans, os.path.join(WEIGHTS_DIR, "bovw_vocab.pkl"))
    joblib.dump(svm,    os.path.join(WEIGHTS_DIR, "bovw_svm.pkl"))
    joblib.dump(scaler, os.path.join(WEIGHTS_DIR, "bovw_scaler.pkl"))
    print(f"\nSaved: bovw_vocab.pkl, bovw_svm.pkl, bovw_scaler.pkl -> {WEIGHTS_DIR}/")