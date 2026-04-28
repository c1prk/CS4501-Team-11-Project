"""
dataset.py — Deepfake Dataset Loader
The same frame filename (e.g. 123_456_frame_0000.jpg) exists in multiple
manipulation folders (Deepfakes/, Face2Face/, FaceSwap/, ...). The original
dataset.py used `break` after the first match in os.walk, which meant only
ONE physical image was loaded per filename — silently dropping ~half the
training data. This version mirrors the index+seen_count logic from
classifier.py so every occurrence in a split file maps to a distinct image.

Also adds:
- Optional training-time augmentation
- Faster loading (index-based, no os.walk per sample)
- ImageNet normalization matching torchvision pretrained models
"""
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# --- standard ImageNet stats for pretrained backbones ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size=224):
    """
    Augmentation for training. Everything here is pose/identity-preserving so
    we don't accidentally erase the manipulation signal. No vertical flips
    (faces aren't vertically symmetric in training data); no heavy rotations
    (would crop out face regions); no grayscale (color channels carry GAN
    artifacts). RandomErasing simulates occlusion and is a strong regularizer.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_eval_transform(image_size=224):
    """No augmentation at eval time."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_file_index(processed_dir):
    """
    Walk data/processed/ once and map every jpg filename to ALL of its paths
    plus labels. Mirrors the logic in classifier.py.
    """
    index = {}
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if f.endswith('.jpg'):
                full_path = os.path.join(root, f)
                label = 1 if 'manipulated_frames' in full_path else 0
                index.setdefault(f, []).append((full_path, label))
    return index


class DeepfakeDataset(Dataset):
    """
    Loads samples defined by a split file. Handles the duplicate-filename case
    by matching the Nth occurrence of a filename in the split file to the Nth
    physical path in the index.
    """

    def __init__(self, split_file, processed_dir, transform=None,
                 file_index=None, return_path=False):
        self.transform = transform if transform is not None else get_eval_transform()
        self.return_path = return_path

        # build index once (or accept a pre-built one if the caller already has it)
        if file_index is None:
            file_index = build_file_index(processed_dir)

        # walk split file, matching the Nth filename occurrence to the Nth path
        with open(split_file, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]

        self.samples = []
        seen_count = {}
        for fname in filenames:
            if fname in file_index:
                count = seen_count.get(fname, 0)
                if count < len(file_index[fname]):
                    self.samples.append(file_index[fname][count])
                    seen_count[fname] = count + 1

        n_fake = sum(1 for _, lbl in self.samples if lbl == 1)
        n_real = len(self.samples) - n_fake
        print(f"[Dataset] {os.path.basename(split_file)}: "
              f"{len(self.samples)} samples ({n_real} real, {n_fake} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)
        if self.return_path:
            return img, torch.tensor(label, dtype=torch.long), img_path
        return img, torch.tensor(label, dtype=torch.long)

    def class_counts(self):
        """Returns (n_real, n_fake) for computing class weights."""
        n_fake = sum(1 for _, lbl in self.samples if lbl == 1)
        return len(self.samples) - n_fake, n_fake