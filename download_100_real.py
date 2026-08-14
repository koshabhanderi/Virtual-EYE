from datasets import load_dataset
from pathlib import Path

# ============================================================
# OUTPUT FOLDER
# ============================================================

output_folder = Path(
    r".\ai_face_training\test\val\real"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("DOWNLOADING 100 REAL FACE IMAGES")
print("=" * 60)

print()
print("Connecting to LFW dataset...")

dataset = load_dataset(
    "marcelohaps/lfw",
    split="train",
    streaming=True
)

print("Connected successfully.")
print()

# ============================================================
# SAVE 100 IMAGES
# ============================================================

count = 0

for item in dataset:

    image = item["image"]

    image = image.convert("RGB")

    filename = output_folder / f"real_{count + 1:03d}.jpg"

    image.save(
        filename,
        "JPEG",
        quality=95
    )

    count += 1

    print(f"Saved {count}/100")

    if count >= 100:
        break

# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("DONE!")
print("=" * 60)

print("REAL IMAGES SAVED:", count)
print("LOCATION:", output_folder.resolve())