"""Unit tests for Generative pillar (VAE & DDPM)."""

from kitepy import Generative


def test_generative_vae_and_ddpm(tmp_path):
    vae = Generative("vae_small", img_size=32)
    vae.train("synthetic", epochs=1, batch_size=8, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    samples = vae.sample(n=2)
    assert samples.shape == (2, 3, 32, 32)

    ddpm = Generative("ddpm_small", img_size=32, timesteps=10)
    ddpm.train("synthetic", epochs=1, batch_size=8, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    diff_samples = ddpm.sample(n=2)
    assert diff_samples.shape == (2, 3, 32, 32)
