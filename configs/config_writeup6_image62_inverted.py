"""Self-contained configuration for inverted-polarity Image 62 projections."""

import os
from pathlib import Path

import torchvision.transforms as T

from utils import utils


dir_this = os.path.dirname(os.path.realpath(__file__))
dir_parent = Path(dir_this).parent

config = dict(
    do_logging=True,
    data=dict(
        data_dir=os.path.join(dir_parent, "data", "writeup6_image62_8colors_inverted"),
        batch_size=8,
        num_workers=0,
        shuffle_data_loader=True,
        transform_train=T.Compose(
            (
                T.RandomRotation(180),
                T.RandomVerticalFlip(0.5),
            )
        ),
        transform_test=None,
    ),
    run=dict(
        epochs=200,
        do_validation=False,
        valid_freq=10,
    ),
    model=dict(
        name="vae",
        zdim=64,
        channels=1,
        do_sigmoid=True,
        vanilla=False,
        encoder=dict(
            name="o2_cnn_encoder",
            cnn_dims=[6, 9, 12, 12, 19, 25],
            layer_type="inducedgated_norm",
            N=-3,
        ),
        decoder=dict(
            name="cnn_decoder",
            cnn_dims=[192, 96, 96, 48, 48, 48, 48, 48],
            out_channels=1,
        ),
    ),
    loss=dict(
        beta=0.01,
        recon_loss_type="bce",
        mask=None,
        align_loss=True,
        align_fourier=True,
        do_flip=True,
        rot_steps=2,
        prior_kwargs=dict(prior="standard"),
    ),
    optimizer=dict(lr=1e-3),
    wandb_log_settings=dict(
        wandb_anonymous=True,
        wandb_enbable_cloud_logging=False,
        project="writeup6-image62-8colors-inverted",
        name="o2vae-writeup6-image62-8colors-inverted-200ep",
        group="writeup6-image62-8colors-inverted",
    ),
    logging=dict(
        train_batch_freq=20,
        tags=["writeup6-image62-8colors-inverted"],
        do_progress_bar=True,
        do_checkpoint=True,
        # Save the final epoch even with the original run.py trainer.
        checkpoint_epoch=1,
        print_verbose=1,
    ),
)

config = utils.Bunch.from_nested_dicts(config)
