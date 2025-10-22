# app.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import json
import os
from io import BytesIO
from PIL import Image, ImageOps

app = FastAPI()
MODEL_PATH = "model.0-9digit.keras"
SCALER_PATH = "scaler.json"  # optional: contains {"mean":..., "std":...}

class PredictRequest(BaseModel):
    pixels: list  # list of 784 values (0-255 or already normalized)

# load model once on startup
model = load_model(MODEL_PATH)

# try to load mean/std if saved during training; otherwise fallback to simple 0-255 scaling
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "r", encoding="utf8") as f:
        scaler = json.load(f)
    MEAN = float(scaler.get("mean", 0.0))
    STD = float(scaler.get("std", 1.0))
    USE_ZSCORE = True
else:
    MEAN = 0.0
    STD = 1.0
    USE_ZSCORE = False


# ---------- helper: preprocess uploaded image ----------
def preprocess_image_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Возвращает numpy array shape (1,28,28,1), dtype=float32,
    готовый к подаче в model.predict()
    """
    # загрузка и конвертация в grayscale
    img = Image.open(BytesIO(file_bytes)).convert("L")  # 'L' - grayscale

    # убрать альфа-канал и EXIF ориентацию по необходимости
    img = ImageOps.exif_transpose(img)

    # привести в режим, аналогичный MNIST: вписать в 28x28, сохраняя пропорции
    # ImageOps.fit масштабирует и центрирует; ANTIALIAS обеспечивает сглаживание
    img = ImageOps.fit(img, (28, 28), Image.LANCZOS)

    # преобразовать в numpy
    arr = np.array(img).astype("float32")

    # Простая эвристика: если фон белый (среднее яркость высокое), то инвертируем,
    # чтобы цифра была светлой на тёмном фоне (как в MNIST).
    # Это помогает, если пользователь загрузил белую бумагу с чёрным рисованным числом.
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Нормализация: сохраняем ту же логику, что и в training
    if USE_ZSCORE:
        arr = (arr - MEAN) / (STD + 1e-7)
    else:
        arr = arr / 255.0

    # final reshape to (1,28,28,1)
    arr = arr.reshape((1, 28, 28, 1))
    return arr


# ---------- existing pixel endpoint (unchanged) ----------
@app.post("/predict")
def predict(req: PredictRequest):
    arr = np.array(req.pixels, dtype="float32")
    if arr.size != 28*28:
        return {"error": "expected 784 pixel values (28x28)"}
    x = arr.reshape(1,28,28,1)
    if USE_ZSCORE:
        x = (x - MEAN) / (STD + 1e-7)
    else:
        x = x / 255.0
    pred = model.predict(x)
    label = int(np.argmax(pred, axis=1)[0])
    return {"label": label, "probabilities": pred.tolist()}


# ---------- new file-upload endpoint ----------
@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    # небольшие проверки
    if not file.content_type.startswith("image/"):
        return {"error": "file must be an image (image/png, image/jpeg, ...)"}

    contents = await file.read()

    # защита: ограничение размера (примерно 5 MB)
    max_size = 5 * 1024 * 1024
    if len(contents) > max_size:
        return {"error": f"file too large (max {max_size} bytes)"}

    try:
        x = preprocess_image_bytes(contents)
    except Exception as e:
        return {"error": "failed to process image", "detail": str(e)}

    pred = model.predict(x)
    label = int(np.argmax(pred, axis=1)[0])
    return {
        "filename": file.filename,
        "label": label,
        "probabilities": pred.tolist()
    }


# optional health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
