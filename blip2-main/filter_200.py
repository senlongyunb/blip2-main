"""Build a 200-image Flickr8k subset for the Mini-BLIP2 reproduction."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "data" / "archive"
SOURCE_IMAGES_DIR = SOURCE_DIR / "Images"
SOURCE_CAPTIONS = SOURCE_DIR / "captions.txt"

OUTPUT_DIR = PROJECT_DIR / "data" / "flickr8k_200"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "Images"
OUTPUT_CAPTIONS = OUTPUT_DIR / "captions.txt"

IMAGE_LIMIT = 200


def main() -> None:
    if not SOURCE_CAPTIONS.exists():
        raise FileNotFoundError(f"Missing captions file: {SOURCE_CAPTIONS}")
    if not SOURCE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Missing images directory: {SOURCE_IMAGES_DIR}")

    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    selected_images: list[str] = []
    selected_set: set[str] = set()
    selected_rows: list[dict[str, str]] = []

    with SOURCE_CAPTIONS.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["image", "caption"]:
            raise ValueError(
                f"Expected CSV header ['image', 'caption'], got {reader.fieldnames}"
            )

        for row in reader:
            image_name = row["image"]

            if image_name not in selected_set:
                if len(selected_images) >= IMAGE_LIMIT:
                    continue
                selected_images.append(image_name)
                selected_set.add(image_name)

            selected_rows.append(
                {
                    "image": image_name,
                    "caption": row["caption"],
                }
            )

    missing_images: list[str] = []
    for image_name in selected_images:
        src = SOURCE_IMAGES_DIR / image_name
        dst = OUTPUT_IMAGES_DIR / image_name
        if not src.exists():
            missing_images.append(image_name)
            continue
        shutil.copy2(src, dst)

    if missing_images:
        raise FileNotFoundError(
            "Some selected images are missing from the source directory: "
            + ", ".join(missing_images[:10])
        )

    with OUTPUT_CAPTIONS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption"])
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"Source captions: {SOURCE_CAPTIONS}")
    print(f"Source images: {SOURCE_IMAGES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Selected images: {len(selected_images)}")
    print(f"Caption rows: {len(selected_rows)}")
    print(f"First image: {selected_images[0]}")
    print(f"Last image: {selected_images[-1]}")


if __name__ == "__main__":
    main()
