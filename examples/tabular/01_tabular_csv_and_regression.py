"""Tabular ML: Classification and regression with TabularResNet & TabularMLP."""

import tempfile
import numpy as np
from kitepy import Tabular

with tempfile.TemporaryDirectory() as tmp:
    # 1. Classification
    clf = Tabular("tabular_resnet", task="classification")
    clf.train("synthetic", epochs=2, batch_size=16, checkpoint_dir=tmp, save_every_n_epochs=0)
    print("Classification probs:", clf.predict(np.random.randn(2, 16), return_probs=True).shape)

    # 2. Regression on custom NumPy arrays
    X = np.random.randn(80, 6).astype(np.float32)
    y = (2.0 * X[:, 0] - 0.5 * X[:, 1]).astype(np.float32)
    reg = Tabular("tabular_mlp", task="regression")
    reg.train((X, y), epochs=2, batch_size=16, checkpoint_dir=tmp, save_every_n_epochs=0)
    print("Regression preds:", reg.predict(X[:3]).squeeze().tolist())
