"""Prepare inverted Image 62 projections directly from raw PNGs."""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


COLORS = ("blue", "green", "orange", "pink", "purple", "red", "white", "yellow")
PROJECTIONS_PER_COLOR = 250
IMAGE_SIZE = 256
PROJECTION_NAME = re.compile(
    r".+_2D_projection_(\d{3})_phi\d+\.\d+_theta\d+\.\d+\.png$"
)


def main():
    default_source_root = (
        Path(__file__).resolve().parents[2]
        / "microglia3d_data"
        / "fig"
        / "kevin"
        / "Writeup6_projections_wrl-4-donors"
        / "2436"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_source_root,
        help="Directory containing the Image 62 color projection folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "writeup6_image62_8colors_inverted",
        help="Directory in which to save X_train.sav and y_train.sav.",
    )
    args = parser.parse_args()

    if not args.source_root.is_dir():
        raise FileNotFoundError(
            f"Projection directory does not exist: {args.source_root}"
        )

    images = []
    labels = []
    counts = {}

    for label, color in enumerate(COLORS):
        color_dirs = [
            path
            for path in args.source_root.iterdir()
            if path.is_dir()
            and "Image_62_" in path.name
            and path.name.endswith(f"_{color}")
        ]
        if len(color_dirs) != 1:
            raise ValueError(
                f"Expected exactly one Image 62 directory ending in '_{color}' "
                f"under {args.source_root}, found {len(color_dirs)}."
            )

        numbered_files = []
        for path in color_dirs[0].glob("*.png"):
            match = PROJECTION_NAME.fullmatch(path.name)
            if match:
                numbered_files.append((int(match.group(1)), path))
        numbered_files.sort(key=lambda item: item[0])

        indices = [index for index, _ in numbered_files]
        expected_indices = list(range(1, PROJECTIONS_PER_COLOR + 1))
        if indices != expected_indices:
            mismatches = [
                (expected, found)
                for expected, found in zip(expected_indices, indices)
                if expected != found
            ][:5]
            raise ValueError(
                f"Expected Image 62 {color} projections numbered "
                f"001-{PROJECTIONS_PER_COLOR:03d}, found {len(indices)} files "
                f"with first index mismatches {mismatches} in {color_dirs[0]}."
            )

        for _, path in numbered_files:
            with Image.open(path) as image:
                if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                    raise ValueError(
                        f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} projection, "
                        f"got {image.size} for {path}."
                    )
                grayscale = image.convert("L")
                pixels = 1.0 - np.array(grayscale, dtype=np.float32, copy=True) / 255.0
            images.append(torch.from_numpy(pixels).unsqueeze(0))
            labels.append(label)

        counts[color] = len(numbered_files)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = torch.stack(images)
    targets = torch.tensor(labels, dtype=torch.long)
    torch.save(data, args.output_dir / "X_train.sav")
    torch.save(targets, args.output_dir / "y_train.sav")

    print(
        f"Saved {len(data)} images with shape {tuple(data.shape[1:])} "
        f"to {args.output_dir}"
    )
    print(f"Images per color: {counts}")
    print(f"Color labels: {dict(enumerate(COLORS))}")
    print("Black/white polarity inverted; labels retain their original order.")


if __name__ == "__main__":
    main()
