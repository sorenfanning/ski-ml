import os
import sys
import glob
import json
import torch
from ultralytics import YOLO

# A Colab notebook's earlier cells are not importable modules. When this code is
# pasted into a notebook cell, reuse the symbols defined by the extraction and
# training cells. When executed as a .py file, use the normal src/ imports.
RUNNING_IN_NOTEBOOK = "__file__" not in globals()

if RUNNING_IN_NOTEBOOK:
    required_symbols = (
        "extract_46_features", "SkiTemporalCNN", "NUM_FEATURES",
        "DATASET_DIR", "MODEL_OUT_DIR", "TARGET_COLUMN", "select_temporal_window",
    )
    missing_symbols = [name for name in required_symbols if name not in globals()]
    if missing_symbols:
        raise RuntimeError(
            "Run the pose-extraction cell and the training-model-definition cell first. "
            f"Missing notebook symbols: {', '.join(missing_symbols)}"
        )
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from extraction.pose_extractor import extract_46_features
    from models.train import (
        DATASET_DIR, MODEL_OUT_DIR, SkiTemporalCNN, NUM_FEATURES, TARGET_COLUMN,
        select_temporal_window,
    )

# --- CONFIGURATION ---
MODEL_DIR = MODEL_OUT_DIR
TEST_INPUT = os.path.join(DATASET_DIR, "test_clips")  # Can be a folder path OR a single video file path
CONFIDENCE_THRESHOLD = 0.70  # Flag anything below 70% as "unsure"


class SkiTrickPredictor:
    def __init__(self, model_dir=MODEL_DIR, target=TARGET_COLUMN):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        # 1. Load metadata & class mapping
        meta_path = os.path.join(model_dir, f"{target}_meta.json")
        weights_path = os.path.join(model_dir, f"{target}_cnn.pt")

        if not os.path.exists(meta_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing model or metadata in {model_dir}/ for '{target}'")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("target") != target:
            raise ValueError(f"Metadata target is '{meta.get('target')}', but '{target}' was requested.")
        if meta.get("num_features") != NUM_FEATURES:
            raise ValueError(
                f"Model expects {meta.get('num_features')} features, but this pipeline extracts {NUM_FEATURES}."
            )
        self.temporal_region = meta.get("temporal_region", "full")
        self.window_fraction = float(meta.get("window_fraction", 1.0))

        # JSON converts int keys to strings, so convert back to int
        self.id2label = {int(k): v for k, v in meta["id2label"].items()}

        # 2. Initialize & load trained PyTorch CNN
        self.model = SkiTemporalCNN(
            in_features=NUM_FEATURES,
            num_classes=len(self.id2label)
        ).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)
        # Support the current self-describing checkpoint and older state_dict-only files.
        is_current_checkpoint = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        state_dict = checkpoint["model_state_dict"] if is_current_checkpoint else checkpoint
        if is_current_checkpoint and checkpoint.get("target_column") != target:
            raise ValueError(
                f"Checkpoint target is '{checkpoint.get('target_column')}', but '{target}' was requested."
            )
        if is_current_checkpoint and checkpoint.get("num_features") != NUM_FEATURES:
            raise ValueError(
                f"Checkpoint expects {checkpoint.get('num_features')} features, but this pipeline extracts {NUM_FEATURES}."
            )
        if is_current_checkpoint and "temporal_region" in checkpoint and checkpoint["temporal_region"] != self.temporal_region:
            raise ValueError("Checkpoint and metadata disagree about the temporal input region.")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 3. Load YOLO Pose model once so batch predictions are fast
        # Match the feature extractor's current pretrained pose backbone.
        self.pose_model = YOLO("yolo11l-pose.pt")
        print(f"Loaded '{target}' model from: {model_dir}")
        print(f"Temporal input: {self.temporal_region} ({self.window_fraction:.0%} of clip)")
        print(f"Prediction device: {self.device}")

    def predict_video(self, video_path):
        # Extract 46 features per frame -> shape: (num_frames, 46)
        seq, diagnostics = extract_46_features(video_path, self.pose_model)
        if len(seq) == 0:
            return {
                "file": os.path.basename(video_path),
                "error": "No frames extracted",
                "details": (
                    f"opened={diagnostics['opened']}, codec={diagnostics['codec']}, "
                    f"reported_frames={diagnostics['reported_frame_count']}"
                ),
            }
        if seq.ndim != 2 or seq.shape[1] != NUM_FEATURES:
            return {"file": os.path.basename(video_path), "error": f"Unexpected feature shape: {seq.shape}"}
        seq = select_temporal_window(seq, self.temporal_region, self.window_fraction)

        # Format for PyTorch Conv1d: (batch=1, channels=46, num_frames)
        tensor = torch.tensor(seq.T, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # Convert raw scores to probabilities

        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())
        predicted_label = self.id2label[pred_id]

        # Build full probability breakdown across all classes
        all_scores = {
            self.id2label[i]: round(float(probs[i].item()), 3)
            for i in range(len(self.id2label))
        }

        return {
            "file": os.path.basename(video_path),
            "prediction": predicted_label if confidence >= CONFIDENCE_THRESHOLD else "unsure",
            "top_guess": predicted_label,
            "confidence": round(confidence, 3),
            "scores": all_scores
        }


def run_test(input_path, target=TARGET_COLUMN):
    """Predict one trained manifest target, such as feature_type or stance_in."""
    predictor = SkiTrickPredictor(target=target)

    if os.path.isfile(input_path):
        videos = [input_path]
    else:
        videos = []
        for ext in ("*.mp4", "*.mov", "*.MP4", "*.MOV"):
            videos.extend(glob.glob(os.path.join(input_path, "**", ext), recursive=True))

    if not videos:
        print(f"No videos found at: {input_path}")
        return

    print(f"\nRunning predictions on {len(videos)} unseen video(s)...\n" + "-" * 60)
    for vid in sorted(videos):
        result = predictor.predict_video(vid)
        if "error" in result:
            print(f"[ERROR] {result['file']}: {result['error']} ({result.get('details', '')})")
            continue

        status = result["prediction"].upper()
        conf_pct = f"{result['confidence']:.1%}"
        print(f"Video: {result['file']:<25} | Result: {status:<8} ({conf_pct}) | Breakdown: {result['scores']}")


if __name__ == "__main__" and not RUNNING_IN_NOTEBOOK:
    # Optional: pass a specific video path via terminal, or default to TEST_INPUT folder
    target_path = sys.argv[1] if len(sys.argv) > 1 else TEST_INPUT
    run_test(target_path)
elif RUNNING_IN_NOTEBOOK:
    print("Prediction pipeline loaded. Run run_test(TEST_INPUT) in the next cell to start predictions.")
