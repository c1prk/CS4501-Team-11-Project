import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, auc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../../"))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_COLS = [
    'asymmetry_score', 'num_kp_left', 'num_kp_right',
    'kp_ratio', 'num_good_matches', 'mean_match_dist', 'std_match_dist'
]


def load_split(name):
    df = pd.read_csv(os.path.join(PROJECT_ROOT, f"sift_features_{name}.csv"))
    df['y'] = (df['label'] == 'fake').astype(int)
    return df


if __name__ == "__main__":
    print("=== SIFT Pipeline Evaluation ===\n")

    df_train = load_split("train")
    df_test = load_split("test")

    # prep data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS].values)
    X_test = scaler.transform(df_test[FEATURE_COLS].values)
    y_train = df_train['y'].values
    y_test = df_test['y'].values

    # train with probability=True so we can get ROC
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
              probability=True, random_state=42)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    # --- 1. ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'SIFT+SVM (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — SIFT + SVM')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curve_sift.png"), dpi=150)
    plt.close()

    # --- 2. Confusion matrix ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix — SIFT + SVM')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_sift.png"), dpi=150)
    plt.close()

    # --- 3. Asymmetry distributions by category ---
    plt.figure(figsize=(10, 5))
    for cat in sorted(df_test['category'].unique()):
        subset = df_test[df_test['category'] == cat]
        plt.hist(subset['asymmetry_score'], bins=20, alpha=0.5, label=cat)
    plt.xlabel('Asymmetry Score')
    plt.ylabel('Count')
    plt.title('Asymmetry Score by Category (Test Set)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "asymmetry_distribution.png"), dpi=150)
    plt.close()

    # --- 4. Per-category breakdown ---
    df_test['pred'] = y_pred
    print("Per-Category Accuracy:")
    for cat in sorted(df_test['category'].unique()):
        sub = df_test[df_test['category'] == cat]
        acc = accuracy_score(sub['y'], sub['pred'])
        print(f"  {cat:<22} ({sub['label'].iloc[0]:<4})  acc={acc:.3f}  n={len(sub)}")

    print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"AUC: {roc_auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Real', 'Fake'])}")
    print(f"Figures saved to {FIGURES_DIR}/")