import cv2
import os

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
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done: {video_name} — {saved_count} frames saved")

def main():
    project_root = get_project_root()
    print(f"Project root: {project_root}")

    manipulation_types = [
        "actors",
        "youtube"
    ]

    for manip_type in manipulation_types:
        videos_folder = os.path.join(project_root, "data", "raw", "original_sequences", manip_type, "c23", "videos")
        output_folder = os.path.join(project_root, "data", "processed", "original_frames", manip_type)

        print(f"Output folder: {output_folder}")

        if not os.path.exists(videos_folder):
            print(f"Skipping {manip_type} — folder not found")
            continue

        all_videos = [f for f in os.listdir(videos_folder) if f.endswith(".mp4")]
        selected_videos = all_videos[:200]

        print(f"\nProcessing {len(selected_videos)} videos from {manip_type}...")

        for video_file in selected_videos:
            video_path = os.path.join(videos_folder, video_file)
            extract_frames(video_path, output_folder)

        print(f"{manip_type} complete!")

if __name__ == "__main__":
    main()