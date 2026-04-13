from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def bootstrap_default_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "tiny-iris-logreg.joblib"

    if model_path.exists():
        return model_path

    iris = load_iris()
    x = iris.data
    y = iris.target

    model = LogisticRegression(max_iter=250, multi_class="auto")
    model.fit(x, y)
    joblib.dump(model, model_path)
    return model_path
