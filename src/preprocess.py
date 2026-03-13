import cv2
import os
from collections import defaultdict

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Could not read FPS for {video_path}")
        cap.release()
        return

    frame_interval = int(round(fps))
    os.makedirs(output_folder, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                print(f"Warning: No faces detected in frame {frame_count} of {video_name}")
            else:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, (224, 224))
                frame_filename = os.path.join(output_folder, f"{video_name}_frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, face_resized)
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done: {video_name} — {saved_count} frames saved")


def create_splits(project_root):
    manipulation_types = [
        "DeepFakeDetection",
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures"
    ]
    original_sources = ["actors", "youtube"]

    splits_folder = os.path.join(project_root, "data", "splits")
    os.makedirs(splits_folder, exist_ok=True)
    for split in ["train.txt", "val.txt", "test.txt"]:
        open(os.path.join(splits_folder, split), "w").close()

    folders = []
    for manip_type in manipulation_types:
        folders.append(os.path.join(project_root, "data", "processed", "manipulated_frames", manip_type))
    for source in original_sources:
        folders.append(os.path.join(project_root, "data", "processed", "original_frames", source))

    for folder in folders:
        if not os.path.exists(folder):
            print(f"Skipping {folder} — not found")
            continue

        # Group frames by video name
        video_frames = defaultdict(list)
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".jpg"):
                video_name = filename.split("_frame_")[0]
                video_frames[video_name].append(filename)

        # Sort and slice video names
        video_names = sorted(video_frames.keys())
        train_videos = video_names[:140]
        val_videos   = video_names[140:170]
        test_videos  = video_names[170:200]

        # Write frame filenames to split files
        for split_name, split_videos in [("train.txt", train_videos), ("val.txt", val_videos), ("test.txt", test_videos)]:
            with open(os.path.join(splits_folder, split_name), "a") as f:
                for video in split_videos:
                    for filename in video_frames[video]:
                        f.write(filename + "\n")

        print(f"Split done: {os.path.basename(folder)} — {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test videos")
    
def main():
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    original_sequences = [
        "actors",
        "youtube"
    ]
    manipulation_types = [
        "DeepFakeDetection",
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures"
    ]

    for manip_type in manipulation_types:
        videos_folder = os.path.join(project_root, "data", "raw", "manipulated_sequences", manip_type, "c23", "videos")
        output_folder = os.path.join(project_root, "data", "processed", "manipulated_frames", manip_type)

        print(f"Output folder: {output_folder}")

        if not os.path.exists(videos_folder):
            print(f"Skipping {manip_type} — folder not found")
            continue

        all_videos = sorted([f for f in os.listdir(videos_folder) if f.endswith(".mp4")])
        selected_videos = all_videos[:200]

        print(f"\nProcessing {len(selected_videos)} videos from {manip_type}...")

        for video_file in selected_videos:
            video_path = os.path.join(videos_folder, video_file)
            extract_frames(video_path, output_folder)

        print(f"{manip_type} complete!")
    for original_sequence in original_sequences:
        videos_folder = os.path.join(project_root, "data", "raw", "original_sequences", original_sequence, "c23", "videos")
        output_folder = os.path.join(project_root, "data", "processed", "original_frames", original_sequence)

        print(f"Output folder: {output_folder}")

        if not os.path.exists(videos_folder):
            print(f"Skipping {original_sequence} — folder not found")
            continue

        all_videos = sorted([f for f in os.listdir(videos_folder) if f.endswith(".mp4")])
        selected_videos = all_videos[:200]

        print(f"\nProcessing {len(selected_videos)} videos from {original_sequence}...")

        for video_file in selected_videos:
            video_path = os.path.join(videos_folder, video_file)
            extract_frames(video_path, output_folder)

        print(f"{original_sequence} complete!")
    create_splits(project_root)

if __name__ == "__main__":
    main()