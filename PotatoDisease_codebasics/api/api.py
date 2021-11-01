from io import BytesIO
from os import listdir
from os.path import join

from numpy import array, argmax, max, expand_dims
from tensorflow.keras.models import load_model
from uvicorn import run
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

_MODEL = load_model(join("saved_models", sorted(listdir(join('saved_models')))[-1]))
_CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello World"


def read_file_as_image(data) -> array:
    return array(Image.open(BytesIO(data)))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = expand_dims(image, 0)

    prediction = _MODEL.predict(img_batch)

    predicted_class = _CLASS_NAMES[argmax(prediction[0])]
    confidence = max(prediction[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence),
    }


def run_api():
    run(app, host="localhost", port=8000)