import numpy as np
from pathlib import Path

WEIGHTS_PATH = Path(__file__).parent / "weights.npz"

_weights: dict | None = None


def _load():
    global _weights
    if _weights is None:
        data = np.load(WEIGHTS_PATH)
        _weights = {k: data[k] for k in data.files}
    return _weights


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def forward(x: np.ndarray) -> dict:
    """
    x: float32 array of shape (784,), values in [0, 1]

    Returns:
        prediction  – int, 0-9
        confidence  – float, softmax probability of the top class
        activations – list of 4 lists: [input(784), hidden1(128), hidden2(64), output(10)]
    """
    w = _load()

    a0 = x
    a1 = _relu(w["W1"] @ a0 + w["b1"])
    a2 = _relu(w["W2"] @ a1 + w["b2"])
    a3 = _softmax(w["W3"] @ a2 + w["b3"])

    prediction = int(np.argmax(a3))

    return {
        "prediction": prediction,
        "confidence": float(a3[prediction]),
        "activations": [
            a0.tolist(),
            a1.tolist(),
            a2.tolist(),
            a3.tolist(),
        ],
    }
