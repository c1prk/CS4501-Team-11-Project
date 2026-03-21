import cv2
import numpy as np
import os
import sys

def extract_canny_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Overall edge density
    edge_area = edges.shape[0] * edges.shape[1]
    overall_density = np.count_nonzero(edges) / edge_area

    # Center vs border edge density
    h, w = edges.shape
    center = edges[h//4:3*h//4, w//4:3*w//4]
    center_density = np.count_nonzero(center) / (center.shape[0] * center.shape[1])

    total_edges = np.count_nonzero(edges)
    border_edges = total_edges - np.count_nonzero(center)
    border_pixels = (h * w) - (center.shape[0] * center.shape[1])
    border_density = border_edges / border_pixels

    boundary_ratio = border_density / (center_density + 1e-6)

    return [overall_density, center_density, border_density, boundary_ratio]


def extract_hsv_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    return [
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ]


def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    canny = extract_canny_features(image)  # 4 numbers
    hsv   = extract_hsv_features(image)    # 6 numbers

    return canny + hsv  # 10 numbers total