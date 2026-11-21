"""Text Classification: Classify text sequences in a few lines."""

import tempfile
from kitepy import TextClassifier

with tempfile.TemporaryDirectory() as tmp:
    clf = TextClassifier("tiny_gpt", num_classes=2)
    clf.class_names = ["negative", "positive"]
    clf.train("synthetic", epochs=1, batch_size=8, checkpoint_dir=tmp, save_every_n_epochs=0)

    preds = clf.predict("this framework is remarkably clean", top_k=2)
    print("Predictions:", preds)
