from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POTATO_MODEL = tf.keras.models.load_model("../saved-models/potato_model.keras", compile=False)
BINARY_MODEL = tf.keras.models.load_model("../saved-models/binary_model.keras", compile=False)
TOMATO_DISEASE_MODEL = tf.keras.models.load_model("../saved-models/disease_model.keras", compile=False)

POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
TOMATO_CLASSES = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')  
    image = image.resize((256, 256))  
    image = np.array(image) / 255.0  
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # First check if it's a tomato
    is_tomato = BINARY_MODEL.predict(img_batch)[0][0]
    
    if is_tomato >= 0.5:  # It's a tomato
        predictions = TOMATO_DISEASE_MODEL.predict(img_batch)
        predicted_class = TOMATO_CLASSES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {
            'plant_type': 'tomato',
            'class': predicted_class,
            'confidence': confidence
        }
    else:  # Try potato classification
        predictions = POTATO_MODEL.predict(img_batch)
        predicted_class = POTATO_CLASSES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {
            'plant_type': 'potato',
            'class': predicted_class,
            'confidence': confidence
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)