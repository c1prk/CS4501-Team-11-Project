import cv2
import os

video_path = os.path.join(os.getcwd(), "data", "raw", "manipulated_sequences", "Deepfakes","c23", "videos", "000_003.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps))  # Save one frame per second
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_folder = os.path.join(project_root, "data", "processed", "raw_frames")
        frame_count = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            # Check if the current frame is the one to be saved every n frames
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Saved {frame_filename}")
                saved_count += 1
            
            frame_count += 1

        cap.release()
        