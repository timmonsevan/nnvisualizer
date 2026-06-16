import base64
import io
import numpy as np
from PIL import Image


def preprocess_image(b64_string: str) -> np.ndarray:
    """
    Accepts a base64-encoded PNG from a canvas element.
    Returns a float32 array of shape (784,) preprocessed to match MNIST:
      1. Tight-crop to the bounding box of the drawn pixels
      2. Resize to fit in a 20×20 box (preserving aspect ratio)
      3. Center in a 28×28 frame (4 px padding)
      4. Shift so the pixel center-of-mass lands at (14, 14)
    """
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    img = Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0

    # Canvas draws dark-on-white by default; MNIST is white-on-black
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # 1. Tight crop
    mask_rows = arr.max(axis=1) > 0.1
    mask_cols = arr.max(axis=0) > 0.1
    if not mask_rows.any():
        return np.zeros(784, dtype=np.float32)

    rmin, rmax = np.where(mask_rows)[0][[0, -1]]
    cmin, cmax = np.where(mask_cols)[0][[0, -1]]
    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]

    # 2. Resize to fit inside 20×20, preserving aspect ratio
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.array(
        Image.fromarray((cropped * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.float32,
    ) / 255.0

    # 3. Paste into center of 28×28
    canvas = np.zeros((28, 28), dtype=np.float32)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    # 4. Shift to center of mass
    total = canvas.sum()
    if total > 0:
        cy = float(np.average(np.arange(28), weights=canvas.sum(axis=1)))
        cx = float(np.average(np.arange(28), weights=canvas.sum(axis=0)))
        dy = int(round(14.0 - cy))
        dx = int(round(14.0 - cx))
        canvas = np.roll(canvas, dy, axis=0)
        canvas = np.roll(canvas, dx, axis=1)
        # zero out pixels that wrapped around the edges
        if dy > 0:
            canvas[:dy, :] = 0
        elif dy < 0:
            canvas[dy:, :] = 0
        if dx > 0:
            canvas[:, :dx] = 0
        elif dx < 0:
            canvas[:, dx:] = 0

    return canvas.flatten()
