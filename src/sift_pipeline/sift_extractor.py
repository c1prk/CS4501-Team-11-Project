import cv2
import numpy as np
import os

def detect_textured_points(gray_img):
    """
    Step 1: Find the 'interesting' parts of the face.
    We use Harris Corner Detection because it's good at finding high-contrast 
    areas like the corners of eyes, edges of teeth, and nostrils.
    """
    # 2 is the block size, 3 is the aperture for Sobel, 0.04 is the Harris detector free parameter
    harris_response = cv2.cornerHarris(gray_img, 2, 3, 0.04)
    
    # We tuned the threshold to 0.005 (down from 0.01) so we get more points.
    # More points = better statistical sample for our asymmetry score.
    corners = np.argwhere(harris_response > 0.005 * harris_response.max())
    
    # Convert these coordinates into cv2 KeyPoints so SIFT can use them
    keypoints = [cv2.KeyPoint(float(p[1]), float(p[0]), 1) for p in corners]
    return keypoints

def get_sift_descriptors(gray_img, keypoints):
    """
    Step 2: Describe what the texture looks like at those corners.
    SIFT creates a mathematical 'fingerprint' for each point we found.
    """
    if not keypoints:
        return None
        
    sift = cv2.SIFT_create()
    # We don't need to re-detect keypoints, just compute descriptors for our Harris points
    _, descriptors = sift.compute(gray_img, keypoints)
    return descriptors

def analyze_facial_asymmetry(image_path):
    """
    Step 3: The 'Mirror' Logic.
    We split the image in half, flip the right side, and see if the 
    textures match up with the left side.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mid = w // 2

    # Split and flip
    left_side = gray[:, :mid]
    # We flip the right side horizontally so it 'overlaps' with the left
    right_mirrored = cv2.flip(gray[:, mid:], 1)

    # Get features for the left side
    kp_l = detect_textured_points(left_side)
    des_l = get_sift_descriptors(left_side, kp_l)

    # Get features for the mirrored right side
    kp_r = detect_textured_points(right_mirrored)
    des_r = get_sift_descriptors(right_mirrored, kp_r)

    return des_l, des_r

def get_asymmetry_score(image_path):
    """
    Step 4: The Final Calculation.
    Compare the descriptors from both sides. If they don't match, 
    the asymmetry score goes up.
    """
    features = analyze_facial_asymmetry(image_path)
    if features is None:
        return None
    
    des_l, des_r = features
    # If one side has no detectable texture, we assume it's totally asymmetrical
    if des_l is None or des_r is None:
        return 1.0 

    # FIX: knnMatch(k=2) crashes if either side has fewer than 2 descriptors
    if len(des_l) < 2 or len(des_r) < 2:
        return 1.0
    
    # Use Brute Force Matcher to find similar textures
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_l, des_r, k=2)
    
    # Lowe's Ratio Test: We use 0.85 (tuned for biological faces).
    # This allows for slight natural differences in eyes/skin while still catching big 'fakes'.
    good_matches = [m for m, n in matches if m.distance < 0.85 * n.distance]
    
    # Calculate ratio of matches to the side with the most features
    match_ratio = len(good_matches) / max(len(des_l), len(des_r))
    
    # Score of 1.0 = Totally Fake/Asymmetrical
    # Score of 0.0 = Perfectly Symmetrical
    return 1.0 - match_ratio