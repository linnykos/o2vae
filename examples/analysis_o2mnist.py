"""
Convert of analysis_o2mnist.ipynb to a standalone script.
Loads a pretrained O2VAE model for MNIST and saves all plots to a directory.
"""

import sys
sys.path.append("../")
import os
import torch
import matplotlib.pyplot as plt
import umap
from torchvision.utils import make_grid

from pathlib import Path
sys.path.append("/gscratch/kzlinlab/projects/microglia3d/git/o2vae_kevin/")

import run
from utils import utils, eval_utils, cluster_utils
from utils import plotting_utils
from configs.config_o2mnist import config

# --- Configuration ---
MODEL_PATH = "/gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin/model_final.pt"
DATA_DIR = "/gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin"
PLOTS_DIR = "/gscratch/kzlinlab/projects/microglia3d/git/o2vae_kevin/examples/example_results"
os.makedirs(PLOTS_DIR, exist_ok=True)

config.data.data_dir = DATA_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# --- Datasets ---
dset, loader, dset_test, loader_test = run.get_datasets_from_config(config)
data_imgs = dset.tensors[0]

f = utils.plot_sample_data(loader)[0]
save(f, "sample_train_data.png")

# --- Build and load model ---
config.model.encoder.n_channels = dset[0][0].shape[0]
model = run.build_model_from_config(config)

print(f"Loading pretrained model from {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)
state_dict = {k: v.clone() for k, v in checkpoint["model_state_dict"].items()}
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
assert all("_basisexpansion" in k for k in missing_keys)
assert len(unexpected_keys) == 0
model.eval().to(device)

# --- Reconstructions ---
x, y = next(iter(loader_test))
reconstruct_grid = eval_utils.reconstruction_grid(model, x, align=False)
reconstruct_grid_aligned = eval_utils.reconstruction_grid(model, x, align=True)

f, axs = plt.subplots(1, 2, figsize=(10, 10))
axs[0].imshow(reconstruct_grid)
axs[1].imshow(reconstruct_grid_aligned)
axs[0].set_title("Reconstructions")
axs[1].set_title("Aligned reconstructions")
axs[0].set_axis_off()
axs[1].set_axis_off()
save(f, "reconstructions.png")

# --- Orientation invariance ---
f, axs = plt.subplots(1, 10, figsize=(20, 20))
for i in range(10):
    xs = eval_utils.rotated_flipped_xs(x[[i]], 90)[:, 0]
    grid = eval_utils.reconstruction_grid(model, xs, align=False, ncol=2)
    axs[i].imshow(grid)
    axs[i].set_axis_off()
save(f, "orientation_invariance.png")

# --- Extract embeddings ---
embeddings, y_true = utils.get_model_embeddings_from_loader(model, loader, return_labels=True)
embeddings_test, y_true_test = utils.get_model_embeddings_from_loader(model, loader_test, return_labels=True)

# --- k-NN image samples ---
first_n = 10000
dist_train = torch.cdist(embeddings[:first_n].to(device), embeddings[:first_n].to(device), p=2).cpu()
dist_argsort = torch.argsort(dist_train.to(device), dim=1).cpu()

k = 10
test_idxs = torch.arange(20)
this_dist_argsort = dist_argsort[test_idxs, : k + 1]
imgs = data_imgs[this_dist_argsort.flatten()]
grid = make_grid(imgs, k + 1, pad_value=0.5).moveaxis(0, 2)
f, ax = plt.subplots(figsize=(12, 12))
ax.imshow(grid)
ax.set_axis_off()
save(f, "knn_samples.png")

# --- UMAP ---
reducer = umap.UMAP()
umap_embedding = reducer.fit_transform(embeddings.cpu().numpy())

f, axs = plotting_utils.plot_embedding_space_w_labels(umap_embedding, y_true)
save(f, "umap_labels.png")

images = loader.dataset.tensors[0]
grid, idxs = plotting_utils.get_embedding_space_embedded_images(umap_embedding, images, n_ximgs=50, n_yimgs=50)
f, ax = plt.subplots(figsize=(12, 12))
ax.set_axis_off()
ax.imshow(grid, cmap="gray")
save(f, "umap_images.png")

# --- Clustering ---
n_clusters = 10
(labels_gmm, labels_kmeans), ((pca, cls_gmm), cls_kmeans), (centers_gmm, centers_kmeans), (scores_gmm, scores_kmeans) \
    = cluster_utils.do_clusterering(embeddings, n_clusters=n_clusters)

labels, centers, scores = labels_gmm, centers_gmm, scores_gmm

grid, counts = cluster_utils.make_sample_grid_for_clustering(
    labels, data_imgs, scores, n_examples=20, stds_filt=1, method="std", paper_figure_grid=0, verbose=0
)
f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(grid)
ax.set_axis_off()
save(f, "cluster_samples.png")

print("\nPurity score for clustering with k=10")
print(f"GMM:    {cluster_utils.purity_score(y_true, labels_gmm):.3f}")
print(f"Kmeans: {cluster_utils.purity_score(y_true, labels_kmeans):.3f}")

print(f"\nAll plots saved to {PLOTS_DIR}")
