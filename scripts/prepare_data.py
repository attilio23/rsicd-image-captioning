import json
from pathlib import Path

from datasets import load_dataset


DATA_ROOT = Path("data/rsicd")
IMAGES_ROOT = DATA_ROOT / "images"
CAPTIONS_ROOT = DATA_ROOT / "captions"

HF_DATASET = "arampacha/rsicd"
HF_CACHE = "./data/_hf_cache"


def export_split(split_name, split_data, images_dir, captions_file):
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in split_data:
        # Strip any directory prefix the dataset may carry in the filename.
        filename = Path(row["filename"]).name
        image = row["image"].convert("RGB")
        image.save(images_dir / filename, format="JPEG", quality=95)
        records.append({"filename": filename, "captions": list(row["captions"])})

    with open(captions_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"  [{split_name}] saved {len(records)} images and captions")


def main():
    print(f"Loading dataset '{HF_DATASET}' from the Hugging Face Hub...")
    raw = load_dataset(HF_DATASET, cache_dir=HF_CACHE)

    # Report filename overlaps across splits (for transparency).
    names = {split: {row["filename"] for row in raw[split]} for split in raw}
    print("Filename overlaps across splits:")
    print(f"  train/valid: {len(names['train'] & names['valid'])}")
    print(f"  train/test:  {len(names['train'] & names['test'])}")
    print(f"  valid/test:  {len(names['valid'] & names['test'])}")

    CAPTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Exporting splits to {DATA_ROOT.resolve()}")
    for split in ["train", "valid", "test"]:
        export_split(
            split_name=split,
            split_data=raw[split],
            images_dir=IMAGES_ROOT / split,
            captions_file=CAPTIONS_ROOT / f"{split}.json",
        )

    print("Done.")


if __name__ == "__main__":
    main()