#!/usr/bin/env python3
import os
import random
import cv2
import json
import dlib
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
import multiprocessing
from imutils import face_utils
from functools import partial
from deepfake.constants import FACE_PREDICTOR_NAME

logger = logging.getLogger(__name__)

# ====== Configuration ======
NUM_FRAMES = 10  

# ====== Helper Functions ======
def parse_video_path(videos_path: str, dataset: str):
    if dataset in ["Actors", "Youtube"]:
        dataset_path = f'{videos_path}/original/{dataset}/'
    else:
        dataset_path = f'{videos_path}/manipulated/{dataset}/'

    movies_path_list = sorted(glob(dataset_path + '*.mp4'))

    if len(movies_path_list) > 0:
        logger.info(f"{len(movies_path_list)} videos found in {dataset}")

    return movies_path_list

def parse_labels(video_path):
    return 0 if "original" in video_path.lower() else 1

def get_output_dir(label, save_images_path):
    return os.path.join(save_images_path, 'real' if label == 0 else 'fake')

def preprocess_video(video_path, save_images_path, face_detector, face_predictor):
    label = parse_labels(video_path)
    save_dir = get_output_dir(label, save_images_path)
    os.makedirs(save_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < NUM_FRAMES:
        logger.warning(f"Video {video_name} has only {frame_count} frames, less than {NUM_FRAMES}")
        frame_idxs = np.arange(0, frame_count)
    else:
        frame_idxs = np.linspace(0, frame_count - 1, NUM_FRAMES, endpoint=True, dtype=int)

    video_meta_dict = {}

    for cnt_frame in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Frame read error at frame {cnt_frame}: {video_path}")
            continue
        if cnt_frame not in frame_idxs:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector(rgb_frame, 1)

        if len(faces) == 0:
            logger.warning(f"No faces in frame {cnt_frame} of {video_name}")
            continue

        landmarks = []
        face_sizes = []

        for face in faces:
            shape = face_predictor(rgb_frame, face)
            shape_np = face_utils.shape_to_np(shape)
            x0, y0 = shape_np[:, 0].min(), shape_np[:, 1].min()
            x1, y1 = shape_np[:, 0].max(), shape_np[:, 1].max()
            face_area = (x1 - x0) * (y1 - y0)
            face_sizes.append(face_area)
            landmarks.append(shape_np)

        if not face_sizes:
            continue

        landmarks = np.array(landmarks)
        largest_face_idx = np.argmax(face_sizes)
        chosen_landmark = landmarks[largest_face_idx]

        # Save image as: videoName_frameNumber.png
        img_filename = f"{video_name}_frame_{cnt_frame}.png"
        img_path = os.path.join(save_dir, img_filename)
        meta_key = os.path.join(os.path.basename(save_images_path), os.path.basename(save_dir), img_filename)

        # Save frame
        cv2.imwrite(img_path, frame)

        # Collect metadata
        video_meta_dict[meta_key] = {
            "landmark": chosen_landmark.tolist(),
            "label": label
        }

    cap.release()
    return video_meta_dict

# ====== Extract Frames ======

def process_video(video_path, save_images_path, face_detector, face_predictor):
    return preprocess_video(str(video_path), str(save_images_path), face_detector, face_predictor)

def extract_frames(config: dict):
    predictor_path = config["modelsdir"] / FACE_PREDICTOR_NAME
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(str(predictor_path))

    datasets = config["datasets"]
    videos_path = config["datadir"]
    save_images_path = config["imgdir"]

    train_size_pcr = config.get("train_size_pcr", 0.8)
    dataset_size = config["dataset_size"]

    train_video_paths_to_process = []
    test_video_paths_to_process = []

    for dataset in datasets:
        all_video_list = parse_video_path(str(videos_path), dataset)
        if len(all_video_list) < dataset_size:
            raise ValueError(\
                f"Requested {dataset_size} videos,"
                f" but only {len(all_video_list)} available in {dataset}"
            )

        video_list = all_video_list[:dataset_size]

        # Split dataset into 80% train and 20% test
        train_size = int(train_size_pcr * len(video_list))
        train_videos = random.sample(video_list, train_size)
        test_videos = [v for v in video_list if v not in train_videos]

        # Process train videos
        train_path = save_images_path / "train"
        os.makedirs(train_path, exist_ok=True)
        for video_path in train_videos:
            train_video_paths_to_process.append((video_path, train_path))

        # Process test videos
        test_path = save_images_path / "test"
        os.makedirs(test_path, exist_ok=True)
        for video_path in test_videos:
            test_video_paths_to_process.append((video_path, test_path))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    process_fn = partial(process_video, face_detector=face_detector, face_predictor=face_predictor)

    train_meta = {}
    test_meta = {}

    # Process TRAIN
    with tqdm(total=len(train_video_paths_to_process), desc="Processing TRAIN videos") as pbar:
        results = pool.starmap(process_fn, train_video_paths_to_process)
        for video_meta in results:
            train_meta.update(video_meta)
            pbar.update(1)

    # Process TEST
    with tqdm(total=len(test_video_paths_to_process), desc="Processing TEST videos") as pbar:
        results = pool.starmap(process_fn, test_video_paths_to_process)
        for video_meta in results:
            test_meta.update(video_meta)
            pbar.update(1)

    pool.close()
    pool.join()

    # Save metadata separately
    with open(os.path.join(str(save_images_path), "train", "ldm.json"), 'w') as f:
        json.dump(train_meta, f, indent=4)

    with open(os.path.join(str(save_images_path), "test", "ldm.json"), 'w') as f:
        json.dump(test_meta, f, indent=4)

    logger.info("Metadata saved: ldm_train.json and ldm_test.json")
