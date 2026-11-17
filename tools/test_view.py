"""Read-only side-by-side viewer for evaluating extracted ski poses.

The left panel shows the source video with its saved pose overlay. The right
panel redraws the same 2D keypoints as a moving skeleton. No labels,
features, pose sidecars, or manifest rows are modified.
"""

import os

import cv2
import numpy as np

from label_viewer import (
    COCO_SKELETON,
    FEATURES_DIR,
    HEAD_KEYPOINT_CONFIDENCE_THRESHOLD,
    JOINT_GROUPS,
    MANIFEST_PATH,
    POSE_COLORS,
    POSE_KEYPOINT_CONFIDENCE_THRESHOLD,
    POSES_DIR,
    RAW_VIDEO_DIR,
    draw_pose_overlay,
    load_manifest,
    load_pose_sidecar,
    normalize_key,
)

WINDOW_NAME = "Ski Pose Test Viewer"
TARGET_FPS = 15.0
PANEL_BACKGROUND = (44, 36, 54)


def pose_sample_at_frame(pose_data, frame_idx):
    """Return the most recent extracted pose for one source-video frame."""
    if pose_data is None:
        return None, None
    sample_idx = np.searchsorted(pose_data["frame_indices"], frame_idx, side="right") - 1
    if sample_idx < 0 or not pose_data["detected"][sample_idx]:
        return None, None
    return pose_data["keypoints_xyn"][sample_idx], pose_data["keypoint_conf"][sample_idx]


def draw_skeleton_panel(shape, pose_data, frame_idx):
    """Render the saved skeleton in the source video's screen coordinates."""
    height, width = shape[:2]
    panel = np.full((height, width, 3), PANEL_BACKGROUND, dtype=np.uint8)
    keypoints, confidences = pose_sample_at_frame(pose_data, frame_idx)

    cv2.putText(panel, "COCO-matched 2D skeleton", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(
        panel,
        "Skull, ribcage, pelvis, and bones driven by saved pose points",
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
    )
    if keypoints is None:
        cv2.putText(panel, "No skier pose", (18, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 2)
        return panel

    def is_visible(index, threshold=POSE_KEYPOINT_CONFIDENCE_THRESHOLD):
        confidence = confidences[index]
        return np.isfinite(keypoints[index]).all() and (not np.isfinite(confidence) or confidence >= threshold)

    visible_indices = [index for index in range(len(keypoints)) if is_visible(index)]
    if not visible_indices:
        cv2.putText(panel, "Pose confidence too low", (18, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 2)
        return panel

    def point(index):
        # Keep the extracted normalized frame position. Unlike the prior
        # body-centred view, this preserves hip travel and camera-relative
        # motion, making the animated skeleton move with the video.
        return tuple(np.round(keypoints[index] * (width, height)).astype(int))

    bone_width = max(5, int(min(width, height) * 0.025))
    bone_outline = (68, 61, 53)
    bone_shadow = (112, 104, 91)
    bone_fill = (205, 200, 181)
    bone_highlight = (239, 236, 219)

    def draw_bone(start_index, end_index, width_multiplier=1.0):
        """Draw a shaded bone exactly between the COCO joints it represents."""
        if not (is_visible(start_index) and is_visible(end_index)):
            return
        start, end = point(start_index), point(end_index)
        thickness = max(3, int(bone_width * width_multiplier))
        cv2.line(panel, start, end, bone_outline, thickness + 4, cv2.LINE_AA)
        cv2.line(panel, start, end, bone_shadow, thickness + 1, cv2.LINE_AA)
        cv2.line(panel, start, end, bone_fill, thickness, cv2.LINE_AA)
        cv2.line(panel, start, end, bone_highlight, max(1, thickness // 3), cv2.LINE_AA)

    # Arms and legs are direct joint-to-joint mappings from the COCO schema.
    for start, end in ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)):
        draw_bone(start, end)

    # Build the torso from the shoulder and hip pairs. The spine and ribs are
    # constrained by the same four COCO points, so the body rotates with the
    # skier instead of remaining upright in the panel.
    shoulder_pair_visible = is_visible(5) and is_visible(6)
    hip_pair_visible = is_visible(11) and is_visible(12)
    shoulder_center = None
    hip_center = None
    torso_axis = None
    if shoulder_pair_visible and hip_pair_visible:
        shoulder_center = (np.array(point(5), dtype=np.float32) + np.array(point(6), dtype=np.float32)) / 2.0
        hip_center = (np.array(point(11), dtype=np.float32) + np.array(point(12), dtype=np.float32)) / 2.0
        torso_vector = hip_center - shoulder_center
        torso_length = float(np.linalg.norm(torso_vector))
        if torso_length > 1:
            torso_axis = torso_vector / torso_length
            torso_perpendicular = np.array((-torso_axis[1], torso_axis[0]))
            draw_bone(5, 6, 1.25)
            draw_bone(11, 12, 1.35)
            cv2.line(panel, tuple(shoulder_center.astype(int)), tuple(hip_center.astype(int)), bone_outline, bone_width + 4, cv2.LINE_AA)
            cv2.line(panel, tuple(shoulder_center.astype(int)), tuple(hip_center.astype(int)), bone_fill, bone_width, cv2.LINE_AA)
            for fraction in (0.20, 0.34, 0.48, 0.62):
                rib_center = shoulder_center + torso_vector * fraction
                rib_half_width = torso_length * (0.30 - fraction * 0.12)
                rib_start = tuple(np.round(rib_center - torso_perpendicular * rib_half_width).astype(int))
                rib_end = tuple(np.round(rib_center + torso_perpendicular * rib_half_width).astype(int))
                cv2.line(panel, rib_start, rib_end, bone_outline, max(3, bone_width - 1), cv2.LINE_AA)
                cv2.line(panel, rib_start, rib_end, bone_fill, max(2, bone_width - 3), cv2.LINE_AA)
            pelvis_axes = (max(12, int(torso_length * 0.25)), max(8, int(torso_length * 0.14)))
            cv2.ellipse(panel, tuple(hip_center.astype(int)), pelvis_axes, 0, 0, 360, bone_outline, -1, cv2.LINE_AA)
            cv2.ellipse(panel, tuple(hip_center.astype(int)), pelvis_axes, 0, 0, 360, bone_fill, max(2, bone_width // 2), cv2.LINE_AA)

    # COCO has wrists and ankles but no hand or foot points. Use simple end
    # caps there; they make the anatomy legible without pretending we measured
    # fingers or ski boots from data we do not have.
    for index in (9, 10, 15, 16):
        if is_visible(index):
            cv2.circle(panel, point(index), bone_width, bone_outline, -1, cv2.LINE_AA)
            cv2.circle(panel, point(index), max(2, bone_width - 3), bone_fill, -1, cv2.LINE_AA)

    # Keep the same helmet/goggle-safe head rule as the overlay: use the nose
    # only when confident; otherwise extrapolate upward from the torso.
    if is_visible(0, HEAD_KEYPOINT_CONFIDENCE_THRESHOLD):
        head = keypoints[0]
    elif all(is_visible(index) for index in (5, 6, 11, 12)):
        shoulders = (keypoints[5] + keypoints[6]) / 2.0
        hips = (keypoints[11] + keypoints[12]) / 2.0
        head = shoulders + 0.35 * (shoulders - hips)
    else:
        head = None
    if head is not None:
        head_point = np.round(head * (width, height)).astype(np.float32)
        skull_radius = max(16, int(min(width, height) * 0.065))
        if shoulder_center is not None:
            neck_vector = head_point - shoulder_center
            neck_length = np.linalg.norm(neck_vector)
            if neck_length > 1:
                skull_center = head_point + neck_vector / neck_length * skull_radius * 0.32
                cv2.line(panel, tuple(shoulder_center.astype(int)), tuple(skull_center.astype(int)), bone_outline, bone_width + 3, cv2.LINE_AA)
                cv2.line(panel, tuple(shoulder_center.astype(int)), tuple(skull_center.astype(int)), bone_fill, bone_width - 1, cv2.LINE_AA)
            else:
                skull_center = head_point
        else:
            skull_center = head_point
        skull_center = tuple(np.round(skull_center).astype(int))
        cv2.circle(panel, skull_center, skull_radius + 3, bone_outline, -1, cv2.LINE_AA)
        cv2.circle(panel, skull_center, skull_radius, bone_fill, -1, cv2.LINE_AA)
        # The saved nose point sits at the bottom half of the skull, rather
        # than being treated as the skull's centre.
        cv2.circle(panel, tuple(head_point.astype(int)), max(3, skull_radius // 7), bone_shadow, -1, cv2.LINE_AA)

    return panel


def draw_footer(frame, clip_id, clip_number, total_clips):
    footer = f"{clip_number}/{total_clips}: {clip_id}    Space: pause  L: slow-mo  A: replay  N/Enter: next  Q: quit"
    cv2.putText(frame, footer, (15, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def resize_for_display(frame, max_height=720):
    height, width = frame.shape[:2]
    if height <= max_height:
        return frame
    scale = max_height / height
    return cv2.resize(frame, (int(width * scale), max_height))


def play_pose_tests():
    """Play every manifest clip with no annotation or disk writes."""
    if not os.path.exists(MANIFEST_PATH):
        print(f"data_manifest.csv not found at: {MANIFEST_PATH}")
        return

    rows, _ = load_manifest()
    print(f"Read-only test viewer: {len(rows)} manifest clips")
    print("No labels or dataset files will be changed.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for row_number, row in enumerate(rows, start=1):
        video_path = os.path.join(RAW_VIDEO_DIR, row["raw_video_rel_path"])
        pose_filename = row.get("pose_file") or f"{row['clip_id']}.npz"
        pose_data = load_pose_sidecar(os.path.join(POSES_DIR, pose_filename))
        if not os.path.exists(video_path):
            print(f"[Warning] Missing video: {video_path}")
            continue
        if pose_data is None:
            print(f"[Warning] Missing or invalid pose sidecar for: {row['clip_id']}")

        replay = True
        while replay:
            replay = False
            paused = False
            slow_mo = False
            frame_idx = 0
            cap = cv2.VideoCapture(video_path)
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_step = max(1, int(round(orig_fps / TARGET_FPS)))

            while cap.isOpened():
                if paused:
                    key = normalize_key(cv2.waitKey(30) & 0xFF)
                    if key == ord(" "):
                        paused = False
                    elif key == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key in (10, 13, ord("n")):
                        break
                    elif key == ord("a"):
                        replay = True
                        break
                    continue

                ret, source_frame = cap.read()
                if not ret:
                    break

                source_frame = resize_for_display(source_frame)
                draw_pose_overlay(source_frame, pose_data, frame_idx)
                skeleton_panel = draw_skeleton_panel(source_frame.shape, pose_data, frame_idx)
                combined = np.hstack((source_frame, skeleton_panel))
                draw_footer(combined, row["clip_id"], row_number, len(rows))
                cv2.imshow(WINDOW_NAME, combined)

                key = normalize_key(cv2.waitKey(65 if slow_mo else 25) & 0xFF)
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                if key in (10, 13, ord("n")):
                    break
                if key == ord("a"):
                    replay = True
                    break
                if key == ord(" "):
                    paused = True
                if key == ord("l"):
                    slow_mo = not slow_mo

                frame_idx += 1

            cap.release()

    cv2.destroyAllWindows()
    print("Finished read-only pose test viewer.")


if __name__ == "__main__":
    play_pose_tests()
