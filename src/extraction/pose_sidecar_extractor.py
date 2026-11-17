"""Extract read-only YOLO pose sidecars for skeleton visualization.

This module is intentionally separate from pose_extractor.py. It never writes
46-feature .npy files and never changes data_manifest.csv; it only writes
poses/<clip_id>.npz files for clips already listed in the manifest.
"""

import csv
import os
from importlib import import_module

import cv2
import numpy as np


RAW_VIDEO_DIR = "/content/drive/MyDrive/Fall 2026 University of Utah/CS 4000/VAULT-APP/ski_ml_shared/raw_videos"
DATASET_DIR = os.path.dirname(RAW_VIDEO_DIR)
POSES_DIR = os.path.join(DATASET_DIR, "poses")
MANIFEST_PATH = os.path.join(DATASET_DIR, "data_manifest.csv")
FAILURE_REPORT_PATH = os.path.join(DATASET_DIR, "pose_extraction_failures.csv")
TARGET_FPS = 15
# Visualization is independent from the 46-feature pipeline, so it can use a
# higher-capacity pose model without changing training data.
POSE_MODEL = "yolo26l-pose.pt"
POSE_SIDECAR_SCHEMA_VERSION = "1.0"


def load_manifest_rows():
    """Read existing clip paths without changing any manifest fields."""
    with open(MANIFEST_PATH, mode="r", newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def extract_pose_sidecar(video_path, pose_model, target_fps=TARGET_FPS):
    """Save the largest detected skier pose per sampled frame, without features."""
    cap = cv2.VideoCapture(video_path)
    diagnostics = {
        "opened": cap.isOpened(),
        "reported_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "reported_fps": cap.get(cv2.CAP_PROP_FPS),
        "decoded_frames": 0,
    }
    if not diagnostics["opened"]:
        cap.release()
        return None, diagnostics

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(source_fps / target_fps)))
    frame_indices, frame_sizes = [], []
    keypoints, confidences, boxes, detected = [], [], [], []
    frame_index = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        diagnostics["decoded_frames"] += 1

        if frame_index % frame_step == 0:
            height, width = frame.shape[:2]
            frame_indices.append(frame_index)
            frame_sizes.append((width, height))
            frame_keypoints = np.full((17, 2), np.nan, dtype=np.float32)
            frame_confidences = np.full(17, np.nan, dtype=np.float32)
            frame_box = np.full(4, np.nan, dtype=np.float32)
            has_pose = False

            results = pose_model(frame, verbose=False)
            if (
                results[0].boxes is not None
                and len(results[0].boxes) > 0
                and results[0].keypoints is not None
            ):
                candidate_boxes = results[0].boxes.xywh.cpu().numpy()
                largest_index = int(np.argmax(candidate_boxes[:, 2] * candidate_boxes[:, 3]))
                candidate_keypoints = results[0].keypoints.xyn[largest_index].cpu().numpy()
                if not np.all(candidate_keypoints == 0):
                    has_pose = True
                    frame_keypoints = candidate_keypoints.astype(np.float32)
                    frame_box = results[0].boxes.xywhn[largest_index].cpu().numpy().astype(np.float32)
                    if results[0].keypoints.conf is not None:
                        frame_confidences = results[0].keypoints.conf[largest_index].cpu().numpy().astype(np.float32)

            keypoints.append(frame_keypoints)
            confidences.append(frame_confidences)
            boxes.append(frame_box)
            detected.append(has_pose)
        frame_index += 1

    cap.release()
    if not frame_indices:
        return None, diagnostics

    return {
        "schema_version": np.array(POSE_SIDECAR_SCHEMA_VERSION),
        "frame_indices": np.array(frame_indices, dtype=np.int32),
        "frame_sizes": np.array(frame_sizes, dtype=np.int32),
        "keypoints_xyn": np.array(keypoints, dtype=np.float32),
        "keypoint_conf": np.array(confidences, dtype=np.float32),
        "bbox_xywhn": np.array(boxes, dtype=np.float32),
        "detected": np.array(detected, dtype=np.bool_),
    }, diagnostics


def save_failures(failures):
    fieldnames = ["clip_id", "raw_video_rel_path", "reason"]
    with open(FAILURE_REPORT_PATH, mode="w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def sync_pose_sidecars(force=False):
    """Generate visualization sidecars for all existing manifest rows.

    `force=True` rebuilds already-present sidecars when you want to test a new
    pose model or visualization extraction setting.
    """
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    try:
        YOLO = import_module("ultralytics").YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install Ultralytics first: pip install ultralytics") from exc

    rows = load_manifest_rows()
    os.makedirs(POSES_DIR, exist_ok=True)
    pose_model = YOLO(POSE_MODEL)
    failures = []
    created = 0
    print(f"Pose sidecar output: {POSES_DIR}")
    print(f"Checking {len(rows)} manifest clips for visualization poses...")

    for row in rows:
        clip_id = row["clip_id"]
        video_path = os.path.join(RAW_VIDEO_DIR, row["raw_video_rel_path"])
        pose_path = os.path.join(POSES_DIR, f"{clip_id}.npz")
        if os.path.exists(pose_path) and not force:
            continue
        if not os.path.exists(video_path):
            failures.append({
                "clip_id": clip_id,
                "raw_video_rel_path": row["raw_video_rel_path"],
                "reason": "Raw video file is missing",
            })
            continue

        print(f"Extracting pose sidecar -> {clip_id}")
        try:
            pose_data, diagnostics = extract_pose_sidecar(video_path, pose_model)
        except Exception as exc:
            failures.append({
                "clip_id": clip_id,
                "raw_video_rel_path": row["raw_video_rel_path"],
                "reason": f"Pose extraction error: {type(exc).__name__}: {exc}",
            })
            continue
        if pose_data is None:
            reason = "Could not decode sampled frames" if diagnostics["opened"] else "Could not open video"
            failures.append({
                "clip_id": clip_id,
                "raw_video_rel_path": row["raw_video_rel_path"],
                "reason": reason,
            })
            continue

        np.savez_compressed(pose_path, **pose_data)
        created += 1

    save_failures(failures)
    print(f"Pose sidecar sync complete: wrote {created} file(s); failures: {len(failures)}")
    print(f"Failure report: {FAILURE_REPORT_PATH}")


if __name__ == "__main__":
    sync_pose_sidecars()
