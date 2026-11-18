"""Generate analysis plots for an inverted Image 62 O2VAE checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from torchvision.utils import make_grid

import run
from configs.config_writeup6_image62_inverted import config
from utils import cluster_utils, eval_utils


DATA_DIR = ROOT / "data" / "writeup6_image62_8colors_inverted"
PLOTS_DIR = ROOT / "examples" / "writeup6_image62_inverted_results"
COLORS = ("blue", "green", "orange", "pink", "purple", "red", "white", "yellow")

device = "cuda" if torch.cuda.is_available() else "cpu"


def save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_embedding_image_grid(embedding, images, n_grid=40):
    """Place the image nearest each 2D embedding-grid cell's center."""
    bounds_min = embedding.min(axis=0)
    bounds_max = embedding.max(axis=0)
    span = np.maximum(bounds_max - bounds_min, np.finfo(np.float32).eps)
    padding = span * 0.01
    bounds_min -= padding
    bounds_max += padding
    span = bounds_max - bounds_min

    cell_indices = np.floor((embedding - bounds_min) / span * n_grid).astype(int)
    cell_indices = np.clip(cell_indices, 0, n_grid - 1)
    image_array = images.squeeze(1).numpy()
    image_height, image_width = image_array.shape[-2:]
    image_grid = np.zeros(
        (n_grid * image_height, n_grid * image_width), dtype=np.float32
    )

    for row in range(n_grid):
        for column in range(n_grid):
            indices = np.flatnonzero(
                (cell_indices[:, 0] == column) & (cell_indices[:, 1] == row)
            )
            if len(indices) == 0:
                continue

            center = bounds_min + (
                np.array([column, row], dtype=np.float32) + 0.5
            ) / n_grid * span
            nearest = indices[
                np.argmin(np.linalg.norm(embedding[indices] - center, axis=1))
            ]
            image_grid[
                row * image_height : (row + 1) * image_height,
                column * image_width : (column + 1) * image_width,
            ] = image_array[nearest]

    return image_grid


def main():
    global DATA_DIR, PLOTS_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Inverted Image 62 checkpoint to analyze (wandb/<run>/files/model.pt).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing X_train.sav and y_train.sav.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory in which to save analysis plots.",
    )
    args = parser.parse_args()

    DATA_DIR = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    PLOTS_DIR = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    plot_colors = tuple("#808080" if color == "white" else color for color in COLORS)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {model_path}")
    if not (DATA_DIR / "X_train.sav").is_file():
        raise FileNotFoundError(
            f"Prepared inverted 256x256 data not found in {DATA_DIR}; run "
            "'python data/prepare_writeup6_image62_inverted.py' first."
        )

    config.data.data_dir = str(DATA_DIR)
    config.data.batch_size = 8
    config.data.shuffle_data_loader = False
    config.data.transform_train = None
    config.data.transform_test = None
    dset, loader, _, _ = run.get_datasets_from_config(config)
    data_imgs = dset.tensors[0]
    y_true = dset.tensors[1].numpy()

    fig, ax = plt.subplots(figsize=(9, 9))
    sample_grid = make_grid(data_imgs[:100], nrow=10, pad_value=0.5).moveaxis(0, 2)
    ax.imshow(sample_grid)
    ax.set_axis_off()
    save(fig, "sample_train_data.png")

    config.model.encoder.n_channels = dset[0][0].shape[0]
    model = run.build_model_from_config(config)
    print(f"Loading trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if "in_indices_" not in key
    }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    unexpected_missing_keys = [
        key
        for key in missing_keys
        if not (
            "in_indices_" in key
            or key.endswith(".filter")
            or key.endswith(".expanded_bias")
        )
    ]
    if unexpected_missing_keys:
        raise RuntimeError(
            f"Unexpected missing model parameters: {unexpected_missing_keys}"
        )
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint parameters: {unexpected_keys}")
    model.eval().to(device)

    x = data_imgs[:32]
    reconstruction_grids = []
    aligned_reconstruction_grids = []
    for image_batch in x.split(config.data.batch_size):
        reconstruction_grids.append(
            eval_utils.reconstruction_grid(
                model, image_batch, align=False, device=device
            )
        )
        aligned_reconstruction_grids.append(
            eval_utils.reconstruction_grid(
                model, image_batch, align=True, device=device
            )
        )
    reconstruction_grid = torch.cat(reconstruction_grids, dim=0)
    reconstruction_grid_aligned = torch.cat(
        aligned_reconstruction_grids, dim=0
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(reconstruction_grid)
    axes[1].imshow(reconstruction_grid_aligned)
    axes[0].set_title("Reconstructions")
    axes[1].set_title("Aligned reconstructions")
    for ax in axes:
        ax.set_axis_off()
    save(fig, "reconstructions.png")

    orientation_indices = np.linspace(0, len(data_imgs) - 1, 10, dtype=int)
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    for index, ax in zip(orientation_indices, axes):
        rotated_images = eval_utils.rotated_flipped_xs(
            data_imgs[index : index + 1], 90
        )[:, 0]
        grid = eval_utils.reconstruction_grid(
            model, rotated_images, align=False, ncol=2, device=device
        )
        ax.imshow(grid)
        ax.set_axis_off()
    save(fig, "orientation_invariance.png")

    # Keep device handling local so no changes to shared utils are needed.
    with torch.no_grad():
        embeddings = torch.cat(
            [model.embed_data(batch[0].to(device)).cpu() for batch in loader]
        )

    neighbor_count = 10
    distances = torch.cdist(embeddings, embeddings)
    distances.fill_diagonal_(float("inf"))
    nearest_indices = torch.argsort(distances, dim=1)[:20, :neighbor_count]
    anchor_indices = torch.arange(len(nearest_indices)).unsqueeze(1)
    knn_indices = torch.cat((anchor_indices, nearest_indices), dim=1)
    neighbor_grid = make_grid(
        data_imgs[knn_indices.flatten()],
        nrow=neighbor_count + 1,
        pad_value=0.5,
    ).moveaxis(0, 2)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(neighbor_grid)
    ax.set_axis_off()
    save(fig, "knn_samples.png")

    umap_embedding = umap.UMAP(random_state=0).fit_transform(embeddings.numpy())
    fig, ax = plt.subplots(figsize=(9, 9))
    for label, color, color_name in zip(range(len(COLORS)), plot_colors, COLORS):
        selected = y_true == label
        ax.scatter(
            umap_embedding[selected, 0],
            umap_embedding[selected, 1],
            color=color,
            s=8,
            label=color_name,
        )
    ax.legend()
    ax.set(xticks=[], yticks=[])
    save(fig, "umap_labels.png")

    embedded_images = make_embedding_image_grid(
        umap_embedding, data_imgs, n_grid=40
    )
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(embedded_images, cmap="gray")
    ax.set_axis_off()
    save(fig, "umap_images.png")

    n_clusters = len(COLORS)
    (labels_gmm, labels_kmeans), _, _, (scores_gmm, _) = (
        cluster_utils.do_clusterering(embeddings, n_clusters=n_clusters)
    )
    cluster_grid, _ = cluster_utils.make_sample_grid_for_clustering(
        labels_gmm,
        data_imgs,
        scores_gmm,
        n_examples=20,
        method="top",
        verbose=0,
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cluster_grid)
    ax.set_axis_off()
    save(fig, "cluster_samples.png")

    print(f"\nPurity score for clustering with k={n_clusters}")
    print(f"GMM:    {cluster_utils.purity_score(y_true, labels_gmm):.3f}")
    print(f"Kmeans: {cluster_utils.purity_score(y_true, labels_kmeans):.3f}")
    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
