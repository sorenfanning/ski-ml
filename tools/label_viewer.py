import os
import csv
import glob
import cv2
import numpy as np

DATASET_RELATIVE_PATH = os.path.join(
    "Fall 2026 University of Utah", "CS 4000", "VAULT-APP", "ski_ml_shared"
)



def find_dataset_dir():
    """Find the locally synced Google Drive copy of the shared dataset."""
    configured_path = os.environ.get("SKI_ML_DATASET_DIR")
    candidates = [configured_path] if configured_path else []
    candidates.append(os.path.expanduser(f"~/Google Drive/My Drive/{DATASET_RELATIVE_PATH}"))
    candidates.extend(
        os.path.join(drive_root, DATASET_RELATIVE_PATH)
        for drive_root in glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive"))
    )

    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "raw_videos")):
            return candidate

    # Return the usual location so the startup error shows the expected path.
    return candidates[0] if configured_path else candidates[1]


DATASET_DIR = find_dataset_dir()
RAW_VIDEO_DIR = os.path.join(DATASET_DIR, "raw_videos")
FEATURES_DIR = os.path.join(DATASET_DIR, "features")
POSES_DIR = os.path.join(DATASET_DIR, "poses")
MANIFEST_PATH = os.path.join(DATASET_DIR, "data_manifest.csv")
WINDOW_NAME = "Ski Trick Viewer & Labeler"
POSE_KEYPOINT_CONFIDENCE_THRESHOLD = 0.45
HEAD_KEYPOINT_CONFIDENCE_THRESHOLD = 0.60
MISSING_LABELS = {"", "null", "none", None}
CORE_LABELS = ("feature_type", "stance_in", "stance_out")
JUMP_LABELS = ("jump_spin", "jump_flip")
RAIL_LABELS = ("rail_entry", "rail_spin_on")
# Colors use OpenCV's BGR order.
POSE_COLORS = {
    "head": (0, 255, 255),       # yellow
    "torso": (255, 255, 0),      # cyan
    "upper_arm": (255, 128, 0),  # blue
    "lower_arm": (255, 0, 128),  # purple
    "upper_leg": (0, 255, 0),    # green
    "lower_leg": (0, 165, 255),  # orange
}
COCO_SKELETON = (
    (5, 6, "torso"), (5, 11, "torso"), (6, 12, "torso"), (11, 12, "torso"),
    (5, 7, "upper_arm"), (6, 8, "upper_arm"),
    (7, 9, "lower_arm"), (8, 10, "lower_arm"),
    (11, 13, "upper_leg"), (12, 14, "upper_leg"),
    (13, 15, "lower_leg"), (14, 16, "lower_leg"),
)
JOINT_GROUPS = {
    "head": {0},  # Draw only the nose; eyes/ears are unreliable under helmets and goggles.
    "torso": {11, 12},
    "upper_arm": {5, 6, 7, 8},
    "lower_arm": {9, 10},
    "upper_leg": {13, 14},
    "lower_leg": {15, 16},
}


def load_manifest():
    with open(MANIFEST_PATH, mode="r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []
    return rows, fieldnames


def save_manifest(rows, fieldnames):
    with open(MANIFEST_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_key(key):
    """Normalize OpenCV key codes so uppercase and lowercase controls both work."""
    if ord("A") <= key <= ord("Z"):
        return key + (ord("a") - ord("A"))
    return key


def is_missing(value):
    return value in MISSING_LABELS


def needs_core_annotation(row):
    """Queue clips missing feature type, stance-in, or stance-out labels."""
    return any(is_missing(row.get(field)) for field in CORE_LABELS)


def set_feature_type(row, feature_type, initials):
    """Set type and mark incompatible detail fields as not applicable."""
    previous_type = row.get("feature_type")
    row["feature_type"] = feature_type
    row["labeled_by"] = initials

    if feature_type == "jump":
        if previous_type != "jump":
            for field in JUMP_LABELS:
                if row.get(field) == "not_applicable":
                    row[field] = "null"
        for field in RAIL_LABELS:
            row[field] = "not_applicable"
    else:
        if previous_type != "rail":
            for field in RAIL_LABELS:
                if row.get(field) == "not_applicable":
                    row[field] = "null"
        for field in JUMP_LABELS:
            row[field] = "not_applicable"


def set_stance(row, field, value, initials):
    row[field] = value
    row["labeled_by"] = initials


def load_pose_sidecar(pose_path):
    """Load a compressed extraction pose sidecar, if available."""
    if not os.path.exists(pose_path):
        return None
    with np.load(pose_path, allow_pickle=False) as pose_file:
        required = {"frame_indices", "keypoints_xyn", "keypoint_conf", "bbox_xywhn", "detected"}
        if not required.issubset(pose_file.files):
            print(f"[Warning] Invalid pose sidecar: {pose_path}")
            return None
        return {name: pose_file[name] for name in required}


def draw_pose_overlay(frame, pose_data, frame_idx, confidence_threshold=POSE_KEYPOINT_CONFIDENCE_THRESHOLD):
    """Draw the nearest sampled YOLO pose on a source video frame."""
    if pose_data is None:
        return
    sample_idx = np.searchsorted(pose_data["frame_indices"], frame_idx, side="right") - 1
    if sample_idx < 0 or not pose_data["detected"][sample_idx]:
        return

    h, w = frame.shape[:2]
    keypoints = pose_data["keypoints_xyn"][sample_idx]
    confidences = pose_data["keypoint_conf"][sample_idx]

    def is_visible(index):
        confidence = confidences[index]
        return np.isfinite(keypoints[index]).all() and (not np.isfinite(confidence) or confidence >= confidence_threshold)

    def point(index):
        return tuple(np.round(keypoints[index] * (w, h)).astype(int))

    def confident_point(index, threshold):
        confidence = confidences[index]
        if not np.isfinite(keypoints[index]).all():
            return None
        if np.isfinite(confidence) and confidence < threshold:
            return None
        return keypoints[index]

    def head_anchor():
        """Use nose when visible; otherwise estimate head position from the torso."""
        nose = confident_point(0, HEAD_KEYPOINT_CONFIDENCE_THRESHOLD)
        if nose is not None:
            return nose

        left_shoulder = confident_point(5, confidence_threshold)
        right_shoulder = confident_point(6, confidence_threshold)
        left_hip = confident_point(11, confidence_threshold)
        right_hip = confident_point(12, confidence_threshold)
        if any(value is None for value in (left_shoulder, right_shoulder, left_hip, right_hip)):
            return None

        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        hip_center = (left_hip + right_hip) / 2.0
        # Extend the hip-to-shoulder torso direction slightly beyond the shoulders.
        return shoulder_center + 0.35 * (shoulder_center - hip_center)

    for start, end, group in COCO_SKELETON:
        if is_visible(start) and is_visible(end):
            cv2.line(frame, point(start), point(end), POSE_COLORS[group], 2)
    for index in range(len(keypoints)):
        if index in {1, 2, 3, 4}:  # Hide COCO eyes and ears in the skier overlay.
            continue
        if is_visible(index):
            if index != 0:
                group = next(group for group, joints in JOINT_GROUPS.items() if index in joints)
                cv2.circle(frame, point(index), 3, POSE_COLORS[group], -1)

    anchor = head_anchor()
    if anchor is not None:
        head_point = tuple(np.round(anchor * (w, h)).astype(int))
        cv2.circle(frame, head_point, 4, POSE_COLORS["head"], -1)


def apply_control(row, key, initials, paused, slow_mo):
    """Apply one keyboard control and return (action, paused, slow_mo)."""
    if key == ord(" "):
        return None, not paused, slow_mo
    if key == ord("l"):
        return None, paused, not slow_mo
    if key == ord("a"):
        return "replay", paused, slow_mo
    if key == ord("q"):
        return "quit", paused, slow_mo
    if key in (10, 13):
        return "next", paused, slow_mo
    if key == ord("n"):
        return "next", paused, slow_mo
    if key == ord("j"):
        set_feature_type(row, "jump", initials)
    elif key == ord("r"):
        set_feature_type(row, "rail", initials)
    elif key == ord("f"):
        set_stance(row, "stance_in", "forward", initials)
    elif key == ord("s"):
        set_stance(row, "stance_in", "switch", initials)
    elif key == ord("g"):
        set_stance(row, "stance_out", "forward", initials)
    elif key == ord("h"):
        set_stance(row, "stance_out", "switch", initials)
    return None, paused, slow_mo


def draw_hud(frame, row, features, feat_idx):
    """Render the current multi-attribute annotation state over the video frame."""
    lines = [
        f"Clip: {row['clip_id']}",
        f"Type: {row.get('feature_type', 'null')}  |  Stance in: {row.get('stance_in', 'null')}  |  Stance out: {row.get('stance_out', 'null')}",
    ]
    if feat_idx >= 0:
        inv_val = features[feat_idx][44]
        ang_vel = features[feat_idx][40]
        lines.append(f"Inversion(Y): {inv_val:+.2f} | SpinVel: {ang_vel:+.2f}")

    for index, text in enumerate(lines):
        color = (0, 255, 255) if index == 1 else (255, 255, 255)
        cv2.putText(frame, text, (15, 30 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    controls = "J/R: type  F/S: in stance  G/H: out stance  Enter: save & next"
    cv2.putText(frame, controls, (15, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame, "N: skip  Space: pause  L: slow-mo  A: replay  Q: save & quit",
                (15, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def play_and_label(filter_incomplete_only=True, initials="team"):
    if not os.path.exists(MANIFEST_PATH):
        print(f"data_manifest.csv not found at: {MANIFEST_PATH}")
        print("Confirm Google Drive for desktop has synced the shared dataset, then run pose_extractor.py if needed.")
        return

    rows, fieldnames = load_manifest()
    print(f"Using shared Drive dataset: {DATASET_DIR}")
    print("\n--- MULTI-ATTRIBUTE CONTROLS ---")
    print("  [J] Jump  | [R] Rail       | [F] In Forward | [S] In Switch")
    print("  [G] Out Forward | [H] Out Switch | [ENTER] Save & Next")
    print("  [SPACE] Pause | [L] Slow-Motion | [A] Replay | [N] Skip | [Q] Save & Quit\n")
    print("Click the video window once before using its keyboard controls.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for row in rows:
        if filter_incomplete_only and not needs_core_annotation(row):
            continue

        vid_path = os.path.join(RAW_VIDEO_DIR, row["raw_video_rel_path"])
        feat_path = os.path.join(FEATURES_DIR, row["feature_file"])
        pose_filename = row.get("pose_file") or f"{row['clip_id']}.npz"
        pose_path = os.path.join(POSES_DIR, pose_filename)

        if not os.path.exists(vid_path):
            print(f"Missing video file: {vid_path}")
            continue

        features = np.load(feat_path) if os.path.exists(feat_path) else None
        pose_data = load_pose_sidecar(pose_path)
        slow_mo = False
        paused = False

        while True:
            cap = cv2.VideoCapture(vid_path)
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_step = max(1, int(round(orig_fps / 15.0)))
            frame_idx = 0
            action = None

            while cap.isOpened():
                if paused:
                    key = normalize_key(cv2.waitKey(30) & 0xFF)
                    action, paused, slow_mo = apply_control(row, key, initials, paused, slow_mo)
                    if action:
                        break
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                # Resize large 4K phone videos for comfortable viewing on Mac screen
                h, w = frame.shape[:2]
                if h > 800:
                    scale = 800 / h
                    frame = cv2.resize(frame, (int(w * scale), 800))

                feat_idx = min(frame_idx // frame_step, len(features) - 1) if features is not None else -1
                draw_pose_overlay(frame, pose_data, frame_idx)
                draw_hud(frame, row, features, feat_idx)
                cv2.imshow(WINDOW_NAME, frame)
                delay = 65 if slow_mo else 25
                key = normalize_key(cv2.waitKey(delay) & 0xFF)
                action, paused, slow_mo = apply_control(row, key, initials, paused, slow_mo)
                if action:
                    break

                frame_idx += 1

            cap.release()

            if action == "replay":
                paused = False
                continue
            elif action == "next":
                save_manifest(rows, fieldnames)
                break
            elif action == "quit":
                save_manifest(rows, fieldnames)
                cv2.destroyAllWindows()
                print("Progress saved to data_manifest.csv.")
                return
            # Loop video automatically until user picks Enter, N, or Q.

    cv2.destroyAllWindows()
    print("All clips reviewed and saved!")


if __name__ == "__main__":
    user_initials = input("Enter your initials (e.g., jd): ").strip() or "team"
    show_all = input("Review ALL clips or only INCOMPLETE core labels? (all/incomplete) [incomplete]: ").strip().lower()
    play_and_label(filter_incomplete_only=(show_all != "all"), initials=user_initials)
