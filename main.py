import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model('fashion_model.keras')

classes = ['Футболка/топ', 'Брюки', 'Свитер', 'Платье', 'Пальто',
           'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки']

app = FastAPI(title="Fashion-MNIST Classifier")

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')

    img = img.point(lambda p: 255 if p > 150 else 0)

    pixels = np.asarray(img, dtype=np.float32)
    if np.mean(pixels) > 127:
        img = Image.fromarray(255 - pixels.astype(np.uint8))

    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    pixels = np.asarray(img, dtype=np.float32) / 255.0
    return pixels[None, ..., None]

@app.get("/")
async def main():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = prepare_image(image_bytes)

    probabilities = model.predict(x, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return {
        "filename": file.filename,
        "predicted_class": classes[predicted_class],
        "class_index": predicted_class,
        "confidence": f"{confidence:.2%}"
    }