import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def get_sift_descriptors(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

def build_visual_vocabulary(all_descriptors, k=200):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2000, n_init=3)
    kmeans.fit(all_descriptors)
    return kmeans

def create_feature_histogram(descriptors, kmeans, k):
    histogram = np.zeros(k)
    if descriptors is not None:
        words = kmeans.predict(descriptors)
        for word in words:
            histogram[word] += 1
    sum_val = np.sum(histogram)
    if sum_val > 0:
        histogram /= sum_val
    return histogram

def train_sift_baseline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    test_split = os.path.join(project_root, "data", "splits", "test.txt")
    processed_dir = os.path.join(project_root, "data", "processed")
    if not os.path.exists(test_split):
        print(f"Error: Missing split file at {test_split}")
        return
    with open(test_split, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]
    all_descriptors = []
    image_data = []
    
    for filename in filenames:
        for root, _, files in os.walk(processed_dir):
            if filename in files:
                path = os.path.join(root, filename)
                desc = get_sift_descriptors(path)
                label = 1 if 'manipulated' in path else 0
                if desc is not None:
                    indices = np.random.choice(len(desc), min(len(desc), 100), replace=False)
                    all_descriptors.extend(desc[indices])
                    image_data.append((desc, label))
                break

    # Build Vocabulary
    k = 200 
    kmeans = build_visual_vocabulary(np.array(all_descriptors), k=k)

    # Create histograms for each image
    X = [create_feature_histogram(d, kmeans, k) for d, l in image_data]
    y = [l for d, l in image_data]

    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='rbf', C=1.0, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['Original', 'Manipulated']))

if __name__ == "__main__":
    train_sift_baseline()