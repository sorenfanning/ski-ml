# ski-ml

Pose-based machine-learning pipeline for classifying freestyle ski clips. The project extracts 46 temporal pose features from video, stores them in a shared dataset, supports collaborative annotation, and trains separate classifiers for trick type and rider stance.

## Pipeline

```text
raw video
  -> YOLO pose extraction
  -> 46-feature .npy sequence + data_manifest.csv
  -> local video labeling
  -> Colab model training
  -> prediction on unseen clips
```

The repository contains code and configuration only. Raw videos, extracted features, live manifests, and trained models are intentionally kept in shared Google Drive and ignored by Git.

## Repository layout

```text
configs/feature_schema.yaml      46-feature schema
src/extraction/pose_extractor.py Extract features and maintain the manifest
tools/label_viewer.py            Local OpenCV annotation viewer
src/models/train.py              CNN training
src/pipeline/predict.py          Inference on unseen clips
requirements.txt                 Python dependencies
```

## Shared dataset layout

Both Colab and local tools expect this Drive structure:

```text
ski_ml_shared/
  raw_videos/
  features/
  poses/
  data_manifest.csv
  extraction_failures.csv
  saved_models/
  test_clips/
```

The default path is:

```text
Google Drive/Fall 2026 University of Utah/CS 4000/VAULT-APP/ski_ml_shared/
```

Set `SKI_ML_DATASET_DIR` to override this location. On Colab, mount Google Drive before running any extraction, training, or prediction code.

## Installation

For local labeling or local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Colab commonly includes PyTorch already; install any missing dependencies with:

```python
!pip install -q -r "/path/to/ski-ml/requirements.txt"
```

## 1. Extract features

Run the feature extractor in Colab after mounting Drive. It scans `raw_videos/` recursively, samples each decodable video at 15 FPS, and writes one `(frames, 46)` NumPy array per clip.

```python
sync_dataset()
```

Unreadable clips are skipped and documented in `extraction_failures.csv`. Phone videos encoded with unsupported codecs may need conversion to H.264 before extraction.

To generate visualization poses separately, run `sync_pose_sidecars()` from `src/extraction/pose_sidecar_extractor.py`. It creates `poses/<clip_id>.npz` files containing sampled frame indices, normalized YOLO keypoints and confidences, largest-person bounding boxes, and detection flags. It never changes the feature `.npy` files or `data_manifest.csv`.

## 2. Label clips locally

Run the viewer on a computer with a locally synced copy of the shared Drive folder:

```bash
python tools/label_viewer.py
```

The viewer annotates several core fields in one pass and preserves existing labels.
When a pose sidecar is available, it also renders the sampled YOLO skeleton over the source video.

| Key | Annotation |
| --- | --- |
| `J` / `R` | Feature type: jump / rail |
| `F` / `S` | Stance-in: forward / switch |
| `G` / `H` | Stance-out: forward / switch |
| `Enter` | Save current labels and advance |
| `N` | Skip without changing labels |
| `Space` | Pause/resume |
| `L` | Toggle slow motion |
| `A` | Replay clip |
| `Q` | Save and quit |

The default review mode shows clips with one or more missing core labels: `feature_type`, `stance_in`, or `stance_out`. Use `all` mode only when correcting existing annotations.

`null` means a value is not labeled. `not_applicable` is assigned automatically to type-specific fields that do not apply—for example, rail details on a jump clip.

## 3. Train models in Colab

The training script creates a separate model per manifest column:

```python
train("feature_type")
train("stance_in")
train("stance_out")
```

Models and metadata are written to `saved_models/`:

```text
feature_type_cnn.pt   feature_type_meta.json
stance_in_cnn.pt      stance_in_meta.json
stance_out_cnn.pt     stance_out_meta.json
```

The stance models use a temporal crop to match the label semantics:

- `stance_in`: first 35% of each clip
- `stance_out`: final 35% of each clip
- `feature_type`: full clip

Each class needs at least two labeled examples for the stratified train/validation split. In practice, collect a balanced and substantially larger dataset before relying on validation accuracy.

## 4. Predict unseen clips

Place new videos in `test_clips/`. In the Colab notebook workflow, first run the extraction and training-definition cells, then run the `predict.py` cell. Finally call:

```python
run_test(TEST_INPUT, target="feature_type")
run_test(TEST_INPUT, target="stance_in")
run_test(TEST_INPUT, target="stance_out")
```

Prediction prints the top class, confidence, and per-class probabilities. A confidence below the configured threshold is reported as `unsure`.

## Feature schema

Every sampled frame has 46 features, documented in [configs/feature_schema.yaml](configs/feature_schema.yaml):

- 34 hip-centered, scale-normalized COCO keypoint coordinates
- hip position and velocity
- shoulder and hip orientation/angular velocity
- ankle separation and orientation proxies
- head-to-hip inversion and horizontal-offset proxies

## Current limitations

- Stance is difficult with 2D body pose alone because the current representation does not directly observe ski tip/tail direction.
- The current feature-type detector chooses the largest detected person as the skier.
- Camera motion, occlusion, unusual viewpoints, and phone-video codecs can reduce extraction quality.
- Do not run multiple labelers against the live manifest at the same time; it is a single shared CSV and concurrent saves can overwrite annotations.

## Data and model versioning

Do not commit `raw_videos/`, `features/`, `data_manifest.csv`, or `saved_models/` to Git. Keep those artifacts in the shared Drive dataset and commit only reproducible code, configuration, and documentation.
