"""
Training script for O2-MNIST on Hyak.
Usage:
    python train_o2mnist_hyak.py \
        --data-dir /gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin \
        --out-dir  /gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin \
        --epochs 41
"""
import argparse
import os
import sys
from pathlib import Path

# Allow importing from the repo root (one level up from examples/)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.transforms as T
import train_loops
import run
from configs.config_o2mnist import config

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="/gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin",
                    help="Directory containing X_train.sav, y_train.sav, etc.")
parser.add_argument("--out-dir", default="/gscratch/kzlinlab/projects/microglia3d/out/o2vae_kevin",
                    help="Directory to save checkpoints")
parser.add_argument("--epochs", type=int, default=None,
                    help="Override number of training epochs from config")
args = parser.parse_args()

# Override config paths
config.data.data_dir = args.data_dir

# RandomRotation uses bilinear interpolation which can push pixels slightly
# outside [0,1], causing BCE loss to assert. Clamp after augmentation.
config.data.transform_train = T.Compose([
    T.RandomRotation(180),
    T.RandomVerticalFlip(0.5),
    T.Lambda(lambda x: x.clamp(0.0, 1.0)),
])
if args.epochs is not None:
    config.run.epochs = args.epochs

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load datasets
dset, loader, dset_test, loader_test = run.get_datasets_from_config(config)
print(f"Train size: {len(dset)}  |  Test size: {len(dset_test)}")

# Build model and optimizer
config.model.encoder.n_channels = dset[0][0].shape[0]
model = run.build_model_from_config(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
print(model.model_details())

# Training loop
try:
    for epoch in range(config.run.epochs):
        train_loops.train(epoch, model, loader, optimizer,
                          do_progress_bar=config.logging.do_progress_bar,
                          do_wandb=False, device=device)

        if config.run.do_validation and epoch % config.run.valid_freq == 0:
            train_loops.valid(epoch, model, loader_test,
                              do_progress_bar=config.logging.do_progress_bar,
                              do_wandb=False, device=device)

        if config.logging.do_checkpoint and epoch % config.logging.checkpoint_epoch == 0:
            ckpt_path = out_dir / f"checkpoint_epoch{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

except KeyboardInterrupt:
    print("Interrupted — saving final model.")

# Save final model
final_path = out_dir / "model_final.pt"
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, final_path)
print(f"Final model saved to {final_path}")
