"""Unit tests for Tabular pillar."""

import numpy as np

from kitepy import Tabular


def test_tabular_classification_regression_and_csv(tmp_path):
    tab_cls = Tabular("tabular_resnet", task="classification")
    tab_cls.train("synthetic", epochs=2, batch_size=16, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert tab_cls.is_trained
    assert tab_cls.predict(np.random.randn(3, 16)).shape == (3, 2)

    X = np.random.randn(60, 5).astype(np.float32)
    y = (X[:, 0] * 3.0 + 1.0).astype(np.float32)
    tab_reg = Tabular("tabular_mlp", task="regression")
    tab_reg.train((X, y), epochs=2, batch_size=15, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert tab_reg.config.input_dim == 5
    assert "train_mae" in tab_reg.training_history[-1]

    csv_path = tmp_path / "sample.csv"
    lines = ["f1,f2,cat,label"]
    for i in range(20):
        lines.append(f"{i * 0.1},{i * 0.2},{'A' if i % 2 == 0 else 'B'},{i % 2}")
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    tab_csv = Tabular("tabular_mlp", task="classification")
    tab_csv.train(str(csv_path), epochs=1, batch_size=4, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert tab_csv.class_names == ["0", "1"]

    # CSV containing NaNs and a remainder batch of size 1 (BatchNorm1d singleton safety)
    nan_csv = tmp_path / "dirty_nan.csv"
    nan_csv.write_text(
        "f1,f2,label\n1.0,0.1,0\n,0.2,1\n3.0,,0\n4.0,0.4,1\n5.0,0.5,0\n6.0,0.6,1\n",
        encoding="utf-8",
    )
    tab_nan = Tabular("tabular_mlp", task="classification")
    tab_nan.train(str(nan_csv), epochs=1, batch_size=2, checkpoint_dir=str(tmp_path), save_every_n_epochs=0)
    assert np.isfinite(tab_nan.training_history[-1]["train_loss"])
