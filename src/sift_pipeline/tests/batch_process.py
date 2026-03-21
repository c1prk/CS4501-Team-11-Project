import os
import sys
import cv2
import pandas as pd

#system path fix 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from sift_extractor import get_asymmetry_score

#config
BASE_DIR = os.path.abspath(os.path.join(current_dir, "../../../"))
BASE_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_SPLIT_FILE = os.path.join(BASE_DIR, "data", "splits", "test.txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "asymmetry_results_full.csv")

def run_full_experiment(samples_per_category=50):
    print("Running full asymmetry experiment...")

    if not os.path.exists(TEST_SPLIT_FILE):
        print(f"Error: Split file not found at {TEST_SPLIT_FILE}")
        return

    # Load target filenames
    with open(TEST_SPLIT_FILE, 'r') as f:
        target_files = [line.strip() for line in f.readlines() if line.strip()]

    # Find all subdirectories in our processed data folder
    # This automatically picks up 'original_frames', 'manipulated_frames', etc.
    categories = [d for d in os.listdir(BASE_DATA_DIR) 
                  if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
    
    print(f"Found {len(categories)} data sections: {categories}")

    results = []
    
    for category in categories:
        cat_path = os.path.join(BASE_DATA_DIR, category)
        print(f"\n>>> Processing Section: {category}")
        
        found_count = 0
        for filename in target_files:
            if found_count >= samples_per_category:
                break
                
            # Recursive search to find the file inside nested folders (like youtube/actors)
            img_path = None
            for root, dirs, files in os.walk(cat_path):
                if filename in files:
                    img_path = os.path.join(root, filename)
                    break
            
            if img_path:
                score = get_asymmetry_score(img_path)
                if score is not None:
                    results.append({
                        'filename': filename,
                        'category': category,
                        'score': score
                    })
                    found_count += 1
                    if found_count % 10 == 0:
                        print(f"    {found_count}/{samples_per_category} complete...")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*50)
    print("                FINAL STATS BY SECTION")
    print("="*50)
    # This will show you the average asymmetry for EACH folder found
    print(df.groupby('category')['score'].mean())
    print(f"\nFull results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    # You can increase this number if you want a bigger sample for your final paper!
    run_full_experiment(samples_per_category=50)