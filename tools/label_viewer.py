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
MANIFEST_PATH = os.path.join(DATASET_DIR, "data_manifest.csv")
WINDOW_NAME = "Ski Trick Viewer & Labeler"
MISSING_LABELS = {"", "null", "none", None}
CORE_LABELS = ("feature_type", "stance_in", "stance_out")
JUMP_LABELS = ("jump_spin", "jump_flip")
RAIL_LABELS = ("rail_entry", "rail_spin_on")


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

        if not os.path.exists(vid_path):
            print(f"Missing video file: {vid_path}")
            continue

        features = np.load(feat_path) if os.path.exists(feat_path) else None
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
