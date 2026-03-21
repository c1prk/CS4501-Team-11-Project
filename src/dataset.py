import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# imageNet normalization values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class DeepfakeDataset(Dataset):
    def __init__(self, split_file, processed_dir, transform=transform):
        """
        split_file   — path to train.txt, val.txt, or test.txt
        processed_dir — path to data/processed/
        transform    — image transforms to apply
        """
        self.transform = transform
        self.samples = []
        with open(split_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines() if line.strip()]

        for filename in filenames:
            for root, dirs, files in os.walk(processed_dir):
                if filename in files:
                    full_path = os.path.join(root, filename)
                    if 'manipulated_frames' in full_path:
                        label = 1
                    else:
                        label = 0

                    self.samples.append((full_path, label))
                    break

        print(f"Loaded {len(self.samples)} samples from {split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)