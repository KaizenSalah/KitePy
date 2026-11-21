"""Audio Classification: Train on spectrograms and predict directly from a raw waveform."""

import tempfile
import torch
from kitepy import Audio

with tempfile.TemporaryDirectory() as tmp:
    audio = Audio("audio_cnn", num_classes=10)
    audio.train("synthetic", epochs=1, batch_size=8, checkpoint_dir=tmp, save_every_n_epochs=0)

    # Raw 1D waveform is automatically converted into a log-STFT spectrogram
    raw_waveform = torch.randn(4096)
    logits = audio.predict(raw_waveform)
    print("Predicted class:", logits.argmax(dim=-1).item())
