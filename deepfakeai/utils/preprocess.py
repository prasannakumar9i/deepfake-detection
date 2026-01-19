import cv2
import os

def extract_frames(video_path, save_dir, frame_rate=10):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            cv2.imwrite(os.path.join(save_dir, f"frame_{frame_id}.jpg"), frame)
            frame_id += 1
        count += 1
    cap.release()
