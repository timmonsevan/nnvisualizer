# nnvisualizer

A handwritten digit recognizer with real-time neural network visualization. Draw a digit on a canvas and watch activations flow through a 3-layer MLP trained on MNIST.

## How it works

The backend runs a small multilayer perceptron (784 → 128 → 64 → 10) in pure NumPy. The model was trained offline with PyTorch and the weights are exported to `weights.npz`, so the inference server has no PyTorch dependency.

Each `/predict` call returns:
- **prediction** — the recognized digit (0–9)
- **confidence** — softmax probability of the top class
- **activations** — neuron values for all four layers (input, hidden1, hidden2, output), intended for the frontend visualizer

## Project structure

```
backend/
├── app/
│   ├── main.py          # FastAPI app, /predict endpoint
│   ├── model.py         # NumPy forward pass, weight loader
│   ├── preprocess.py    # Canvas PNG → MNIST-style 784-dim vector
│   └── weights.npz      # Trained weights (committed artifact)
├── training/
│   └── train.py         # Offline PyTorch training script
└── requirements.txt
```

## Running the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`.

## API

### `POST /predict`

**Request body**
```json
{ "image": "<base64-encoded PNG>" }
```

**Response**
```json
{
  "prediction": 7,
  "confidence": 0.982,
  "activations": [
    [/* 784 input values */],
    [/* 128 hidden layer 1 values */],
    [/* 64  hidden layer 2 values */],
    [/* 10  output probabilities */]
  ]
}
```

## Configuration

Set the `ALLOWED_ORIGINS` environment variable to control CORS (comma-separated). Defaults to `https://evantimmons.space,http://localhost:5173`.

```bash
ALLOWED_ORIGINS="http://localhost:5173" uvicorn app.main:app --reload
```

## Retraining

The training script requires PyTorch and torchvision (not in `requirements.txt`).

```bash
pip install torch torchvision
cd backend
python -m training.train
```

This downloads MNIST, trains for 10 epochs with Adam (lr=1e-3, batch=256), and overwrites `app/weights.npz`.

## Preprocessing

Incoming canvas images are preprocessed to match MNIST conventions before inference:

1. Invert if the image is light-on-dark (canvas draws dark-on-white)
2. Tight-crop to the bounding box of the drawn pixels
3. Resize to fit inside a 20×20 box (aspect ratio preserved)
4. Center in a 28×28 frame with 4 px padding
5. Shift so the pixel center-of-mass lands at (14, 14)
