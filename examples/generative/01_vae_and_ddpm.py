"""Generative AI: Train a ConvVAE and sample new images."""

import tempfile
from kitepy import Generative

with tempfile.TemporaryDirectory() as tmp:
    vae = Generative("vae_small", img_size=32)
    vae.train("synthetic", epochs=1, batch_size=8, checkpoint_dir=tmp, save_every_n_epochs=0)

    samples = vae.sample(n=4)
    print("Generated image batch shape:", samples.shape)
