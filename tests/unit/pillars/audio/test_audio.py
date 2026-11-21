"""Unit tests for Audio pillar."""

import torch

from kitepy import Audio


def test_audio_classification_and_waveform_predict(tmp_path):
    audio = Audio("audio_cnn", num_classes=10)
    audio.train("synthetic", epochs=1, batch_size=8, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert audio.is_trained

    # Accepts raw 1D waveform (auto-converted via STFT)
    waveform = torch.randn(4096)
    logits = audio.predict(waveform)
    assert logits.shape == (1, 10)


def test_audio_transformer_variant(tmp_path):
    audio_tr = Audio("audio_transformer", num_classes=5)
    audio_tr.train("synthetic", epochs=1, batch_size=8, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert audio_tr.is_trained
    assert audio_tr.predict(torch.randn(2, 1, 64, 128)).shape == (2, 5)


def test_audio_real_wav_folder_pipeline(tmp_path):
    import wave
    import numpy as np

    audio_root = tmp_path / "wav_dataset"
    for cls_name in ["beep", "noise"]:
        cls_dir = audio_root / cls_name
        cls_dir.mkdir(parents=True)
        for i in range(3):
            wav_path = cls_dir / f"clip_{i}.wav"
            samples = (np.random.randn(8000) * 1000).astype(np.int16)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(samples.tobytes())

    audio = Audio("audio_cnn")
    audio.train(str(audio_root), epochs=1, batch_size=2, checkpoint_dir=str(tmp_path / "ck"), save_every_n_epochs=0)
    assert audio.config.num_classes == 2
    assert audio.class_names == ["beep", "noise"]
