import os
import csv
import json
import random
import glob
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATASET_RELATIVE_PATH = os.path.join(
    "Fall 2026 University of Utah", "CS 4000", "VAULT-APP", "ski_ml_shared"
)


def find_dataset_dir():
    """Find the shared dataset in Colab's mounted Drive or local Drive sync."""
    configured_path = os.environ.get("SKI_ML_DATASET_DIR")
    candidates = [configured_path] if configured_path else []
    candidates.extend([
        os.path.join("/content/drive/MyDrive", DATASET_RELATIVE_PATH),
        os.path.expanduser(f"~/Google Drive/My Drive/{DATASET_RELATIVE_PATH}"),
    ])
    candidates.extend(
        os.path.join(drive_root, DATASET_RELATIVE_PATH)
        for drive_root in glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive"))
    )

    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "data_manifest.csv")):
            return candidate

    expected_path = candidates[0] if configured_path else candidates[1]
    raise FileNotFoundError(
        "Could not find the shared ski dataset. Mount/sync Google Drive or set "
        f"SKI_ML_DATASET_DIR. Expected manifest near: {expected_path}"
    )


DATASET_DIR = find_dataset_dir()
MANIFEST_PATH = os.path.join(DATASET_DIR, "data_manifest.csv")
FEATURES_DIR = os.path.join(DATASET_DIR, "features")
MODEL_OUT_DIR = os.path.join(DATASET_DIR, "saved_models")
TARGET_COLUMN = os.environ.get("SKI_ML_TARGET_COLUMN", "feature_type")

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_FEATURES = 46
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42
STANCE_WINDOW_FRACTION = 0.35


def temporal_region_for_target(target_column):
    """Return the relevant portion of a clip for a manifest target."""
    if target_column == "stance_in":
        return "start"
    if target_column == "stance_out":
        return "end"
    return "full"


def select_temporal_window(seq, region, window_fraction=STANCE_WINDOW_FRACTION):
    """Select a start/end window without changing the frame feature schema."""
    if region == "full":
        return seq
    if region not in {"start", "end"}:
        raise ValueError(f"Unsupported temporal region: {region}")
    if not 0 < window_fraction <= 1:
        raise ValueError(f"window_fraction must be in (0, 1], got {window_fraction}")

    window_length = max(1, math.ceil(len(seq) * window_fraction))
    return seq[:window_length] if region == "start" else seq[-window_length:]


def mirror_46_features(seq):
    """
    Horizontally mirrors a (frames, 46) feature array on the fly during training.
    Doubles effective dataset size without cluttering your features/ folder.
    """
    m = seq.copy()
    # 1. Flip X coordinates of the 17 normalized joints (indices 0, 2, ..., 32)
    m[:, 0:34:2] *= -1.0
    # 2. Mirror global hip_x around screen center (0.5 -> 1.0 - x) and flip vx
    m[:, 34] = 1.0 - m[:, 34]
    m[:, 36] *= -1.0
    # 3. Reflect orientation angles and flip angular velocities
    #    (theta becomes pi - theta under a horizontal mirror).
    m[:, [38, 39, 43]] = (np.pi - m[:, [38, 39, 43]] + np.pi) % (2 * np.pi) - np.pi
    m[:, 40] *= -1.0
    m[:, 41] *= -1.0
    # 4. Flip head-to-hip horizontal offset
    m[:, 45] *= -1.0
    return m


class ManifestSkiDataset(Dataset):
    def __init__(self, samples, augment=False, temporal_region="full", window_fraction=STANCE_WINDOW_FRACTION):
        """
        samples: list of tuples (npy_path, int_label)
        augment: if True, randomly mirrors sequences 50% of the time
        """
        self.samples = samples
        self.augment = augment
        self.temporal_region = temporal_region
        self.window_fraction = window_fraction

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        seq = np.load(npy_path)  # Shape: (num_frames, 46)
        if seq.ndim != 2 or seq.shape[1] != NUM_FEATURES or len(seq) == 0:
            raise ValueError(
                f"Invalid feature array at {npy_path}: expected (frames, {NUM_FEATURES}), got {seq.shape}."
            )
        seq = select_temporal_window(seq, self.temporal_region, self.window_fraction)

        if self.augment and random.random() < 0.5:
            seq = mirror_46_features(seq)

        # Transpose to (channels=46, num_frames) for PyTorch Conv1d
        x_tensor = torch.tensor(seq.T, dtype=torch.float32)
        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor


def pad_collate(batch):
    """Pads variable-length clips (5s to 30s) to the longest clip in the batch."""
    max_len = max(item[0].shape[1] for item in batch)
    padded_seqs, labels = [], []

    for seq, label in batch:
        pad_amount = max_len - seq.shape[1]
        padded = torch.nn.functional.pad(seq, (0, pad_amount))
        padded_seqs.append(padded)
        labels.append(label)

    return torch.stack(padded_seqs), torch.stack(labels)


class SkiTemporalCNN(nn.Module):
    """
    1D-CNN with Global Max Pooling.
    Scans variable-length clips for trick signatures anywhere in the video.
    """
    def __init__(self, in_features=NUM_FEATURES, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def load_samples_from_manifest(manifest_path, features_dir, target_col):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    raw_items = []
    unique_labels = set()

    with open(manifest_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_str = (row.get(target_col) or "").strip().lower()
            # "not_applicable" excludes incompatible clips (for example, rail
            # clips when training jump_spin). "none" remains a valid category
            # for rail_spin_on.
            if label_str in ("", "null", "not_applicable"):
                continue

            npy_path = os.path.join(features_dir, row["feature_file"])
            if not os.path.exists(npy_path):
                print(f"[Warning] Missing feature file: {npy_path}")
                continue

            raw_items.append((npy_path, label_str))
            unique_labels.add(label_str)

    sorted_labels = sorted(unique_labels)
    label2id = {lbl: idx for idx, lbl in enumerate(sorted_labels)}
    id2label = {idx: lbl for idx, lbl in enumerate(sorted_labels)}

    samples = [(path, label2id[lbl]) for path, lbl in raw_items]
    return samples, label2id, id2label


def validate_split_requirements(labels, label2id):
    """Raise a clear error before scikit-learn's stratified split fails."""
    class_counts = {label: labels.count(label_id) for label, label_id in label2id.items()}
    too_small = {label: count for label, count in class_counts.items() if count < 2}
    if too_small:
        raise ValueError(
            "Each class needs at least two labeled clips for a stratified train/validation split. "
            f"Current counts: {class_counts}"
        )

    num_classes = len(label2id)
    validation_size = max(num_classes, math.ceil(len(labels) * VALIDATION_FRACTION))
    if len(labels) - validation_size < num_classes:
        raise ValueError(
            "Not enough labeled clips to keep every class in both train and validation sets. "
            f"Current counts: {class_counts}"
        )
    return validation_size


def train(target_column=TARGET_COLUMN):
    """Train one classifier for a manifest column, such as stance_in or stance_out."""
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Model output: {MODEL_OUT_DIR}")
    samples, label2id, id2label = load_samples_from_manifest(
        MANIFEST_PATH, FEATURES_DIR, target_column
    )

    if len(label2id) < 2:
        print(f"Not enough labeled data yet! Found {len(samples)} clips across classes: {label2id}")
        return

    print(f"Training target: '{target_column}' | Total clips: {len(samples)} | Classes: {label2id}")

    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    validation_size = validate_split_requirements(labels, label2id)
    print(f"Split: {len(samples) - validation_size} train / {validation_size} validation")
    temporal_region = temporal_region_for_target(target_column)
    print(f"Temporal input: {temporal_region} ({STANCE_WINDOW_FRACTION:.0%} of each clip for stance targets)")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    train_p, val_p, train_l, val_l = train_test_split(
        paths, labels, test_size=validation_size, random_state=RANDOM_SEED, stratify=labels
    )

    train_dataset = ManifestSkiDataset(
        list(zip(train_p, train_l)), augment=True, temporal_region=temporal_region
    )
    val_dataset = ManifestSkiDataset(
        list(zip(val_p, val_l)), augment=False, temporal_region=temporal_region
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training device: {device}")
    model = SkiTemporalCNN(in_features=NUM_FEATURES, num_classes=len(label2id)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    weights_path = os.path.join(MODEL_OUT_DIR, f"{target_column}_cnn.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)

        # Validation pass
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = torch.argmax(model(x), dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total if total > 0 else 0.0
        avg_loss = train_loss / len(train_dataset)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "target_column": target_column,
                "label2id": label2id,
                "num_features": NUM_FEATURES,
                "temporal_region": temporal_region,
                "window_fraction": STANCE_WINDOW_FRACTION,
            }, weights_path)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.1%}")

    # Save metadata so your backend knows which class ID maps to which label
    meta_path = os.path.join(MODEL_OUT_DIR, f"{target_column}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "target": target_column,
            "id2label": id2label,
            "num_features": NUM_FEATURES,
            "target_fps": 15,
            "temporal_region": temporal_region,
            "window_fraction": STANCE_WINDOW_FRACTION,
            "best_val_acc": best_val_acc,
        }, f, indent=2)

    print(f"\nDone! Best Val Accuracy: {best_val_acc:.1%}")
    print(f"Saved model weights to: {weights_path}")
    print(f"Saved class mapping to: {meta_path}")


if __name__ == "__main__":
    train()
