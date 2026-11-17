import os
import glob
import csv
from importlib import import_module
import cv2
import numpy as np

# --- CONFIGURATION ---
# This path is inside Colab's mounted Google Drive. Store generated artifacts
# beside raw_videos so they persist after the temporary Colab runtime ends.
RAW_VIDEO_DIR = "/content/drive/MyDrive/Fall 2026 University of Utah/CS 4000/VAULT-APP/ski_ml_shared/raw_videos"  # Update if running locally
DATASET_DIR = os.path.dirname(RAW_VIDEO_DIR)
FEATURES_DIR = os.path.join(DATASET_DIR, "features")
MANIFEST_PATH = os.path.join(DATASET_DIR, "data_manifest.csv")
FAILURE_REPORT_PATH = os.path.join(DATASET_DIR, "extraction_failures.csv")
TARGET_FPS = 15

MANIFEST_COLUMNS = [
    "clip_id",
    "raw_video_rel_path",
    "feature_file",
    "num_frames",
    "fps",
    "feature_type",   # jump | rail
    "stance_in",      # forward | switch
    "stance_out",     # forward | switch
    "jump_spin",      # 0 | 180 | 360 | 540 | 720 | 900 | 1080+
    "jump_flip",      # true | false
    "rail_entry",     # regular | lipslide
    "rail_spin_on",   # none | 270_on | 450_on
    "labeled_by",     # teammate initials
    "notes"
]

# COCO Keypoint Indices used by YOLO Pose
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


def unwrap_angle_diff(curr_angle, prev_angle):
    """Calculates shortest angular difference in radians across the -pi/pi boundary."""
    if prev_angle is None:
        return 0.0
    diff = curr_angle - prev_angle
    return (diff + np.pi) % (2 * np.pi) - np.pi


def fourcc_to_string(fourcc):
    """Convert OpenCV's numeric codec identifier into a readable four-character code."""
    value = int(fourcc)
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).strip() or "unknown"


def extract_46_features(video_path, pose_model, target_fps=TARGET_FPS):
    cap = cv2.VideoCapture(video_path)
    diagnostics = {
        "opened": cap.isOpened(),
        "backend": cap.getBackendName() if cap.isOpened() else "unavailable",
        "reported_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "reported_fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC)),
        "decoded_frames": 0,
    }

    if not diagnostics["opened"]:
        cap.release()
        return np.empty((0, 46), dtype=np.float32), diagnostics

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(orig_fps / target_fps)))

    sequence = []
    prev_hip = None
    prev_shoulder_angle = None
    prev_hip_angle = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        diagnostics["decoded_frames"] += 1

        if frame_idx % frame_step == 0:
            results = pose_model(frame, verbose=False)

            if (
                results[0].boxes is not None
                and len(results[0].boxes) > 0
                and results[0].keypoints is not None
            ):
                # Lock onto largest bounding box (primary skier)
                boxes = results[0].boxes.xywh.cpu().numpy()
                main_idx = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
                kpts = results[0].keypoints.xyn[main_idx].cpu().numpy()  # (17, 2)

                if not np.all(kpts == 0):
                    # 1. Global Trajectory (4 features: hip_x, hip_y, vx, vy)
                    hip_center = (kpts[L_HIP] + kpts[R_HIP]) / 2.0
                    hip_vel = (hip_center - prev_hip) if prev_hip is not None else np.zeros(2)
                    prev_hip = hip_center

                    # 2. Normalized Pose (34 features)
                    centered = kpts - hip_center
                    max_dist = np.max(np.linalg.norm(centered, axis=1))
                    if max_dist > 0:
                        centered = centered / max_dist

                    # 3. Rotational Kinematics (4 features)
                    s_vec = kpts[R_SHOULDER] - kpts[L_SHOULDER]
                    h_vec = kpts[R_HIP] - kpts[L_HIP]
                    shoulder_angle = float(np.arctan2(s_vec[1], s_vec[0]))
                    hip_angle = float(np.arctan2(h_vec[1], h_vec[0]))

                    shoulder_ang_vel = unwrap_angle_diff(shoulder_angle, prev_shoulder_angle)
                    hip_ang_vel = unwrap_angle_diff(hip_angle, prev_hip_angle)
                    prev_shoulder_angle = shoulder_angle
                    prev_hip_angle = hip_angle

                    # 4. Ski/Boot & Inversion Proxies (4 features)
                    ankle_vec = centered[R_ANKLE] - centered[L_ANKLE]
                    ankle_sep = float(np.linalg.norm(ankle_vec))
                    ankle_angle = float(np.arctan2(ankle_vec[1], ankle_vec[0]))
                    head_hip_y = float(centered[NOSE][1])  # >0 means nose is below hips (inverted!)
                    head_hip_x = float(centered[NOSE][0])  # captures looking back in switch stance

                    frame_feat = np.concatenate([
                        centered.flatten(),                                          # [0:34]
                        hip_center,                                                  # [34:36]
                        hip_vel,                                                     # [36:38]
                        [shoulder_angle, hip_angle, shoulder_ang_vel, hip_ang_vel],  # [38:42]
                        [ankle_sep, ankle_angle, head_hip_y, head_hip_x]             # [42:46]
                    ])
                    sequence.append(frame_feat)
                else:
                    sequence.append(np.zeros(46))
            else:
                sequence.append(np.zeros(46))

        frame_idx += 1

    cap.release()
    return np.array(sequence, dtype=np.float32), diagnostics


def load_existing_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["clip_id"]: row for row in reader}


def save_manifest(manifest_path, records_dict):
    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for clip_id in sorted(records_dict.keys()):
            writer.writerow(records_dict[clip_id])


def save_failure_report(failures):
    fieldnames = [
        "clip_id", "raw_video_rel_path", "reason", "file_size_bytes",
        "opened", "backend", "codec", "reported_frame_count", "reported_fps",
        "width", "height", "decoded_frames",
    ]
    with open(FAILURE_REPORT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def sync_dataset():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    print(f"Raw videos: {RAW_VIDEO_DIR}")
    print(f"Feature output: {FEATURES_DIR}")
    print(f"Manifest output: {MANIFEST_PATH}")
    manifest = load_existing_manifest(MANIFEST_PATH)
    try:
        YOLO = import_module("ultralytics").YOLO
    except ModuleNotFoundError as exc:
        if exc.name != "ultralytics":
            raise
        raise RuntimeError(
            "The 'ultralytics' package is required; install it with 'pip install ultralytics'."
        ) from exc
    pose_model = YOLO("yolo11l-pose.pt")
    failures = []

    # Find all .mp4 and .mov files recursively inside RAW_VIDEO_DIR
    video_paths = []
    for ext in ("*.mp4", "*.mov", "*.MP4", "*.MOV"):
        video_paths.extend(glob.glob(os.path.join(RAW_VIDEO_DIR, "**", ext), recursive=True))

    print(f"Found {len(video_paths)} raw videos. Checking for new clips...")
    new_count = 0

    for vid_path in sorted(video_paths):
        clip_id = os.path.splitext(os.path.basename(vid_path))[0]
        feat_filename = f"{clip_id}.npy"
        feat_out_path = os.path.join(FEATURES_DIR, feat_filename)

        # Skip if already extracted and in manifest
        if os.path.exists(feat_out_path) and clip_id in manifest:
            continue

        print(f"Extracting 46 features -> {clip_id}")
        seq, diagnostics = extract_46_features(vid_path, pose_model)
        if len(seq) == 0:
            reason = "OpenCV could not open the video" if not diagnostics["opened"] else "OpenCV could not decode any frames"
            failures.append({
                "clip_id": clip_id,
                "raw_video_rel_path": os.path.relpath(vid_path, RAW_VIDEO_DIR),
                "reason": reason,
                "file_size_bytes": os.path.getsize(vid_path),
                **diagnostics,
            })
            print(
                f"  [Warning] {reason} for {clip_id}; "
                f"codec={diagnostics['codec']}, backend={diagnostics['backend']}, "
                f"reported_frames={diagnostics['reported_frame_count']}."
            )
            continue

        np.save(feat_out_path, seq)

        # Auto-infer initial feature_type if teammate placed video inside a 'jump/' or 'rail/' subfolder
        rel_path = os.path.relpath(vid_path, RAW_VIDEO_DIR)
        lower_rel = rel_path.lower()
        inferred_type = "jump" if "jump" in lower_rel else ("rail" if "rail" in lower_rel else "null")

        if clip_id not in manifest:
            manifest[clip_id] = {
                "clip_id": clip_id,
                "raw_video_rel_path": rel_path,
                "feature_file": feat_filename,
                "num_frames": len(seq),
                "fps": TARGET_FPS,
                "feature_type": inferred_type,
                "stance_in": "null",
                "stance_out": "null",
                "jump_spin": "null",
                "jump_flip": "null",
                "rail_entry": "null",
                "rail_spin_on": "null",
                "labeled_by": "auto" if inferred_type != "null" else "null",
                "notes": ""
            }
            new_count += 1

    save_manifest(MANIFEST_PATH, manifest)
    save_failure_report(failures)
    print(f"Sync complete! Added {new_count} new clips. Total manifest rows: {len(manifest)}")
    print(f"Unreadable-video report: {FAILURE_REPORT_PATH} ({len(failures)} clip(s))")


if __name__ == "__main__":
    sync_dataset()
