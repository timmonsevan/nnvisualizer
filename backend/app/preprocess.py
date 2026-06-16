import base64
import io
import numpy as np
from PIL import Image


def preprocess_image(b64_string: str) -> np.ndarray:
    """
    Accepts a base64-encoded PNG from a canvas element.
    Returns a float32 array of shape (784,) normalized to [0, 1]
    with white digit on black background, matching MNIST conventions.
    """
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    img = Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0

    # Canvas default is dark stroke on white background; MNIST is the opposite.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr.flatten()
