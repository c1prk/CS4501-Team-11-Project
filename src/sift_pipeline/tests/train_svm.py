import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# this file lives at src/sift_pipeline/tests/train_svm.py
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../../"))

FEATURE_COLS = [
    'asymmetry_score',
    'edge_density', 'center_density', 'border_density', 'boundary_ratio',
    'h_mean', 'h_std', 's_mean', 's_std', 'v_mean', 'v_std'
]

def load_split(name):
    path = os.path.join(PROJECT_ROOT, f"sift_features_{name}.csv")
    if not os.path.exists(path):
        print(f"Missing: {path} — run batch_process.py first")
        return None, None
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values
    y = (df['label'] == 'fake').astype(int).values
    print(f"  {name}: {len(df)} samples ({(y==1).sum()} fake, {(y==0).sum()} real)")
    return X, y

if __name__ == "__main__":
    print("=== SIFT + Canny + HSV -> SVM ===\n")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    if any(x is None for x in [X_train, X_val, X_test]):
        exit()

    # normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # train
    print("\nTraining SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)

    # val
    print("\n--- Validation ---")
    y_val_pred = svm.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred, target_names=['Real', 'Fake']))

    # test
    print("--- Test ---")
    y_test_pred = svm.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=['Real', 'Fake']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))