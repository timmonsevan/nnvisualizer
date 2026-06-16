import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.model import forward
from app.preprocess import preprocess_image

app = FastAPI()

_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://evantimmons.space,http://localhost:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image: str  # base64-encoded PNG from canvas


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        x = preprocess_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    return forward(x)
