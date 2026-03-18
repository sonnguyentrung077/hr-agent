"""Preprocess a video or image into avatar data for Wav2Lip inference.

Extracts frames, runs face detection, crops and resizes faces, and saves
the results (full frames, face crops, bounding box coordinates) to disk.

Usage:
    python preprocess_avatar.py --input /path/to/video.mp4 --avatar_id wav2lip256_avatar1
    python preprocess_avatar.py --input /path/to/image.png --avatar_id wav2lip256_avatar1
"""

import argparse
import glob
import os
import pickle

import cv2
import face_detection
import numpy as np
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for face detection.")


def video_to_frames(video_path: str, save_path: str, max_frames: int = 10_000_000):
    """Extract frames from video and save as numbered PNGs."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(save_path, f"{count:08d}.png"), frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames.")


def get_smoothened_boxes(boxes: np.ndarray, T: int = 5) -> np.ndarray:
    """Smooth bounding boxes over a temporal window."""
    for i in range(len(boxes)):
        window = boxes[max(0, i - T // 2) : min(len(boxes), i + T // 2 + 1)]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def detect_faces(
    images: list[np.ndarray],
    pads: tuple[int, int, int, int] = (0, 10, 0, 0),
    batch_size: int = 16,
    smooth: bool = True,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Run face detection on all frames.

    Returns list of (cropped_face, (y1, y2, x1, x2)) tuples.
    """
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device
    )

    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size), desc="Detecting faces"):
                predictions.extend(
                    detector.get_detections_for_batch(np.array(images[i : i + batch_size]))
                )
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    "Image too big to run face detection on GPU. "
                    "Try resizing the video first."
                )
            batch_size //= 2
            print(f"OOM error; reducing batch size to {batch_size}")
            continue
        break

    pady1, pady2, padx1, padx2 = pads
    results = []
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError(
                "Face not detected in a frame. "
                "Ensure the video contains a face in all frames."
            )
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if smooth:
        boxes = get_smoothened_boxes(boxes, T=5)

    output = []
    for image, (x1, y1, x2, y2) in zip(images, boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        output.append((image[y1:y2, x1:x2], (y1, y2, x1, x2)))

    del detector
    return output


def main():
    parser = argparse.ArgumentParser(description="Preprocess a video or image for Wav2Lip avatar inference")
    parser.add_argument("--input", required=True, help="Path to input video or image")
    parser.add_argument("--avatar_id", default="wav2lip256_avatar1", help="Avatar identifier")
    parser.add_argument("--img_size", type=int, default=256, help="Face crop resolution (default: 256)")
    parser.add_argument("--pads", nargs=4, type=int, default=[0, 10, 0, 0],
                        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
                        help="Padding around detected face (default: 0 10 0 0)")
    parser.add_argument("--face_det_batch_size", type=int, default=16,
                        help="Batch size for face detection")
    parser.add_argument("--nosmooth", action="store_true",
                        help="Disable temporal smoothing of face detections")
    parser.add_argument("--output_dir", default="./data/avatars",
                        help="Base output directory (default: ./data/avatars)")
    args = parser.parse_args()

    avatar_path = os.path.join(args.output_dir, args.avatar_id)
    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    face_imgs_path = os.path.join(avatar_path, "face_imgs")
    coords_path = os.path.join(avatar_path, "coords.pkl")

    os.makedirs(full_imgs_path, exist_ok=True)
    os.makedirs(face_imgs_path, exist_ok=True)

    # Step 1: Extract frames (video) or load single image
    input_path = args.input
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    is_image = os.path.splitext(input_path)[1].lower() in image_exts

    if is_image:
        print(f"Loading image from {input_path}...")
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")
        cv2.imwrite(os.path.join(full_imgs_path, "00000000.png"), img)
        frames = [img]
    else:
        print(f"Extracting frames from {input_path}...")
        video_to_frames(input_path, full_imgs_path)
        img_list = sorted(glob.glob(os.path.join(full_imgs_path, "*.[jpJP][pnPN]*[gG]")))
        if not img_list:
            raise FileNotFoundError(f"No frames found in {full_imgs_path}")
        print(f"Loading {len(img_list)} frames...")
        frames = [cv2.imread(p) for p in tqdm(img_list, desc="Reading images")]

    # Step 3: Detect faces
    print("Running face detection...")
    face_results = detect_faces(
        frames,
        pads=tuple(args.pads),
        batch_size=args.face_det_batch_size,
        smooth=not args.nosmooth,
    )

    # Step 4: Save cropped faces and coordinates
    coord_list = []
    for idx, (face_crop, coords) in enumerate(tqdm(face_results, desc="Saving face crops")):
        resized = cv2.resize(face_crop, (args.img_size, args.img_size))
        cv2.imwrite(os.path.join(face_imgs_path, f"{idx:08d}.png"), resized)
        coord_list.append(coords)

    with open(coords_path, "wb") as f:
        pickle.dump(coord_list, f)

    print(f"\nDone! Avatar data saved to {avatar_path}")
    print(f"  Full frames: {len(frames)}")
    print(f"  Face crops:  {len(coord_list)} ({args.img_size}x{args.img_size})")


if __name__ == "__main__":
    main()
