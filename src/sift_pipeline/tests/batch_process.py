import os
import sys
import cv2
import pandas as pd

#system path fix 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from sift_extractor import get_asymmetry_score
from classical_classifier import extract_features

#config
BASE_DIR = os.path.abspath(os.path.join(current_dir, "../../../"))
BASE_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")


def discover_categories():
    """FIX: walk one level into manipulated_frames/ and original_frames/ 
    so we get each type (Deepfakes, Face2Face, etc.) as its own category."""
    categories = {}
    for top_folder, label in [("manipulated_frames", "fake"), ("original_frames", "real")]:
        top_path = os.path.join(BASE_DATA_DIR, top_folder)
        if not os.path.exists(top_path):
            continue
        for sub in sorted(os.listdir(top_path)):
            sub_path = os.path.join(top_path, sub)
            if os.path.isdir(sub_path):
                categories[sub] = {"path": sub_path, "label": label}
    return categories


def find_image(filename, search_dir):
    """Recursive search to find a file inside nested folders."""
    for root, dirs, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def run_full_experiment(samples_per_category=50):
    print("Running full experiment...")

    categories = discover_categories()
    if not categories:
        print("Error: No categories found under data/processed/")
        return
    print(f"Found {len(categories)} categories: {list(categories.keys())}")

    all_dfs = []

    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(SPLITS_DIR, f"{split_name}.txt")
        if not os.path.exists(split_file):
            print(f"Split file not found: {split_file}")
            continue

        with open(split_file, 'r') as f:
            target_files = [line.strip() for line in f.readlines() if line.strip()]

        print(f"\n--- {split_name.upper()} ({len(target_files)} filenames) ---")
        results = []

        for cat_name, cat_info in categories.items():
            cat_path = cat_info["path"]
            label = cat_info["label"]
            print(f"  {cat_name} ({label})...", end=" ")

            found_count = 0
            for filename in target_files:
                if found_count >= samples_per_category:
                    break

                img_path = find_image(filename, cat_path)
                if img_path:
                    # get asymmetry score from sift_extractor
                    asym = get_asymmetry_score(img_path)
                    # get canny + hsv features from classical_classifier
                    extra = extract_features(img_path)

                    if asym is not None and extra is not None:
                        results.append({
                            'filename': filename,
                            'category': cat_name,
                            'label': label,
                            'split': split_name,
                            'asymmetry_score': asym,
                            'edge_density': extra[0],
                            'center_density': extra[1],
                            'border_density': extra[2],
                            'boundary_ratio': extra[3],
                            'h_mean': extra[4], 'h_std': extra[5],
                            's_mean': extra[6], 's_std': extra[7],
                            'v_mean': extra[8], 'v_std': extra[9]
                        })
                        found_count += 1
                        if found_count % 10 == 0:
                            print(f"{found_count}", end=" ")

            print(f"-> {found_count} samples")

        df = pd.DataFrame(results)
        out_path = os.path.join(BASE_DIR, f"sift_features_{split_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(BASE_DIR, "sift_features_all.csv"), index=False)
        print("\n" + "="*50)
        print("         Mean Asymmetry by Category")
        print("="*50)
        print(combined.groupby(['category', 'label'])['asymmetry_score'].mean())


if __name__ == "__main__":
    run_full_experiment(samples_per_category=50)