# Inverted Image 62 pipeline

Run these commands from the repository root with the O2VAE environment active.
This pipeline uses the existing model and trainer, with a self-contained config,
preparation script, and analysis script. No other Writeup configs or scripts are
required.

## Prepare and train

```bash
python data/prepare_writeup6_image62_inverted.py
python run.py configs.config_writeup6_image62_inverted
```

Preparation reads raw PNGs directly from the sibling directory
`../microglia3d_data/fig/kevin/Writeup6_projections_wrl-4-donors/2436`.
To use a different location, pass `--source-root /path/to/2436` to the preparation
command. No previously prepared non-inverted tensors are needed.

Each of the eight Image 62 color folders must contain 250 numbered 256x256
2D projections. Separate 3D render PNGs are excluded. Grayscale pixels are
normalized to [0, 1], then inverted with `1 - pixels`. Labels are ordered blue,
green, orange, pink, purple, red, white, yellow (0 through 7).
The generated `X_train.sav` and `y_train.sav` are saved under
`data/writeup6_image62_8colors_inverted/`.

The config trains a new 256x256 O2VAE for 200 epochs. Checkpoints are saved every
epoch so the original trainer also saves the final epoch. Training prints the
W&B run directory containing `files/model.pt`; logging stays offline.

## Analyze

Replace `<run-folder>` with the inverted training run's directory:

```bash
python examples/analysis_writeup6_image62_inverted.py --model-path "wandb/<run-folder>/files/model.pt"
```

An explicit checkpoint is required to avoid selecting another experiment's
model. Analysis uses the inverted config, dataset, and eight color labels by
default. Optional `--data-dir` and `--output-dir` arguments override the dataset
and plot directories.

Sample images, reconstructions, orientation checks, nearest neighbors, UMAP
plots, and cluster samples are written to
`examples/writeup6_image62_inverted_results/`. Clustering purity is printed to
the terminal. Analysis supports CPU or CUDA through local device handling.

## Files to commit

Commit only these files for this pipeline:

```text
.gitignore
README_writeup6_image62_inverted.md
configs/config_writeup6_image62_inverted.py
data/prepare_writeup6_image62_inverted.py
examples/analysis_writeup6_image62_inverted.py
```

Generated tensors, plots, checkpoints, and W&B run files stay local. The shared
`run.py`, `utils/utils.py`, and existing README files do not need changes for
this pipeline.
