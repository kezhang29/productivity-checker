import os
import shutil
import random
import zipfile
import pandas as pd

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

ZIP_FILE = "human-action-recognition-har-dataset.zip"

EXTRACT_DIR = "/tmp/har_dataset"

SOURCE = os.path.join(EXTRACT_DIR, "Human Action Recognition")

DEST = "/home/nvidia/jetson-inference/python/training/classification/data/classroom"

KEEP_LABELS = {
    "using_laptop",
    "sleeping",
    "texting",
    "calling",
    "eating",
    "listening_to_music",
    "sitting",
}

VAL_SPLIT = 0.15

random.seed(42)

# ---------------------------------------------------
# Extract dataset
# ---------------------------------------------------

if not os.path.exists(SOURCE):
    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall(EXTRACT_DIR)

    print("Done.")

print(f"Dataset found:\n{SOURCE}")

# ---------------------------------------------------
# Create destination folders
# ---------------------------------------------------

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DEST, split), exist_ok=True)

# ---------------------------------------------------
# Read training labels
# ---------------------------------------------------

csv_path = os.path.join(SOURCE, "Training_set.csv")

df = pd.read_csv(csv_path)

print(f"Found {len(df)} labeled images.")

# ---------------------------------------------------
# Copy images
# ---------------------------------------------------

copied = 0

for label in KEEP_LABELS:

    subset = df[df["label"] == label]

    images = subset["filename"].tolist()

    random.shuffle(images)

    val_count = int(len(images) * VAL_SPLIT)

    train_imgs = images[val_count:]
    val_imgs = images[:val_count]

    # train
    train_dir = os.path.join(DEST, "train", label)
    os.makedirs(train_dir, exist_ok=True)

    for img in train_imgs:
        src = os.path.join(SOURCE, "train", img)

        if os.path.exists(src):
            shutil.copy2(src, train_dir)
            copied += 1

    # val
    val_dir = os.path.join(DEST, "val", label)
    os.makedirs(val_dir, exist_ok=True)

    for img in val_imgs:
        src = os.path.join(SOURCE, "train", img)

        if os.path.exists(src):
            shutil.copy2(src, val_dir)
            copied += 1

print(f"\nCopied {copied} images.\n")

print("Dataset created successfully!")

print(f"\nLocation:\n{DEST}")