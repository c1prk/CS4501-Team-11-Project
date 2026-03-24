"""
classifier.py — Bag of Visual Words (BoVW) Pipeline
Place at: src/classifier.py

Extracts SIFT descriptors from face images, clusters them into
a vocabulary of 200 visual words, represents each image as a 
histogram, then trains an SVM to classify real vs fake.
"""
import os
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# --- paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "data", "weights")

VOCAB_SIZE = 200  # number of visual words (clusters)


def build_file_index():
    """Walk data/processed/ once and map every jpg filename to ALL its paths + labels.
    The same filename (e.g. 149_152_frame_0003.jpg) can exist in Deepfakes/,
    Face2Face/, FaceSwap/, etc. — different images, same name."""
    index = {}
    for root, dirs, files in os.walk(PROCESSED_DIR):
        for f in files:
            if f.endswith('.jpg'):
                full_path = os.path.join(root, f)
                label = 1 if 'manipulated_frames' in full_path else 0
                if f not in index:
                    index[f] = []
                index[f].append((full_path, label))
    return index


def load_split(split_name, index):
    """Read a split file and look up each filename in our index.
    Each filename may map to multiple images across different manipulation types.
    The Nth occurrence of a filename in the split file maps to the Nth path in the index."""
    split_file = os.path.join(SPLITS_DIR, f"{split_name}.txt")
    with open(split_file, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    samples = []
    seen_count = {}  # tracks how many times we've seen each filename
    for fname in filenames:
        if fname in index:
            count = seen_count.get(fname, 0)
            if count < len(index[fname]):
                samples.append(index[fname][count])
                seen_count[fname] = count + 1

    labels = [s[1] for s in samples]
    print(f"  {split_name}: {len(samples)} images ({sum(labels)} fake, {len(labels) - sum(labels)} real)")
    return samples


def extract_sift(image_path):
    """Run SIFT detect+compute on a grayscale image. Returns descriptors or None."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img, None)
    return descriptors


def build_vocabulary(samples, max_images=2000):
    """
    Step 1: Collect SIFT descriptors from a sample of training images.
    Step 2: Cluster them with K-Means into VOCAB_SIZE visual words.
    """
    print(f"\nBuilding vocabulary from {min(max_images, len(samples))} images...")

    rng = np.random.RandomState(42)

    # sample a subset so this doesn't take forever
    if len(samples) > max_images:
        idx = rng.choice(len(samples), max_images, replace=False)
        samples = [samples[i] for i in idx]

    all_descriptors = []
    for i, (path, label) in enumerate(samples):
        des = extract_sift(path)
        if des is not None:
            # cap at 100 per image so no single image dominates the vocabulary
            if len(des) > 100:
                idx = rng.choice(len(des), 100, replace=False)
                des = des[idx]
            all_descriptors.append(des)
        if (i + 1) % 100 == 0:
            print(f"  descriptors from {i + 1}/{len(samples)} images...")

    all_descriptors = np.vstack(all_descriptors).astype(np.float32)
    print(f"  total descriptors collected: {len(all_descriptors)}")

    # cap at 100k so K-Means finishes in reasonable time
    if len(all_descriptors) > 100000:
        idx = rng.choice(len(all_descriptors), 100000, replace=False)
        all_descriptors = all_descriptors[idx]
        print(f"  sampled down to {len(all_descriptors)} for clustering")

    print(f"  running K-Means with k={VOCAB_SIZE}...")
    kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, random_state=42, batch_size=1000)
    kmeans.fit(all_descriptors)
    print("  vocabulary built!")
    return kmeans


def image_to_histogram(image_path, kmeans):
    """
    Convert one image into a normalized histogram of visual word frequencies.
    This is the BoVW representation — a fixed-length vector regardless of
    how many keypoints the image has.
    """
    des = extract_sift(image_path)
    hist = np.zeros(VOCAB_SIZE)
    if des is None:
        return hist

    # assign each descriptor to its nearest visual word
    words = kmeans.predict(des.astype(np.float32))
    for w in words:
        hist[w] += 1

    # L1 normalize so images with more keypoints don't dominate
    total = np.sum(hist)
    if total > 0:
        hist /= total
    return hist


def encode_split(samples, kmeans, name=""):
    """Encode all images in a split as histograms."""
    X, y = [], []
    for i, (path, label) in enumerate(samples):
        hist = image_to_histogram(path, kmeans)
        X.append(hist)
        y.append(label)
        if (i + 1) % 500 == 0:
            print(f"  {name}: encoded {i + 1}/{len(samples)}...")
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("=== Bag of Visual Words Pipeline ===\n")

    # step 0: index all processed images (one-time directory walk)
    print("Indexing images...")
    index = build_file_index()
    print(f"  {len(index)} images found\n")

    # step 1: load splits
    train_samples = load_split("train", index)
    val_samples = load_split("val", index)
    test_samples = load_split("test", index)

    # step 2: build vocabulary from training data
    kmeans = build_vocabulary(train_samples)

    # step 3: encode all images as histograms
    print("\nEncoding images...")
    X_train, y_train = encode_split(train_samples, kmeans, "train")
    X_val, y_val = encode_split(val_samples, kmeans, "val")
    X_test, y_test = encode_split(test_samples, kmeans, "test")

    # step 4: normalize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # step 5: train SVM
    print("\nTraining SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)

    # step 6: evaluate
    print("\n--- Validation ---")
    y_val_pred = svm.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred, target_names=['Original', 'Manipulated']))

    print("--- Test ---")
    y_test_pred = svm.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=['Original', 'Manipulated']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # save model + vocabulary
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    joblib.dump(kmeans, os.path.join(WEIGHTS_DIR, "bovw_vocab.pkl"))
    joblib.dump(svm, os.path.join(WEIGHTS_DIR, "bovw_svm.pkl"))
    joblib.dump(scaler, os.path.join(WEIGHTS_DIR, "bovw_scaler.pkl"))
    print(f"\nModel + vocabulary saved to {WEIGHTS_DIR}/")