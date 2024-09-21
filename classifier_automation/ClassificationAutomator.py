import pymongo
from keras import Model
import numpy as np
import bson
from PIL import Image
import requests
from io import BytesIO

class ClassificationAutomator:
    def __init__(self, connection_string, db_name, collection_name, models: dict[str, Model]):
        client = pymongo.MongoClient(connection_string)
        db = client[db_name]
        self.collection_name = collection_name
        self.collection = db[self.collection_name]
        self.models: dict[str, Model] = models

    def evaluate_img(self, image: Image.Image) -> dict[str, float]:
        preds = {}
        for feature, model in self.models.items():
            # Resize image to the correct dimensions
            _, img_h, img_w, _ = model.input_shape
            image = image.resize((img_w, img_h))
            image_arr = np.expand_dims(np.array(image), axis=0)
            prediction = model.predict(image_arr)
            prediction = prediction.tolist()[0]
            preds[feature] = prediction
        return preds

    def convert_image_to_bytes(self, image: Image.Image):
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    def process_and_upload_img(self, image: Image.Image):
        document = {
            "image": self.convert_image_to_bytes(image),
            **self.evaluate_img(image)
        }
        self.collection.insert_one(document)

    def run(self, classification_count = 100):
        for _ in range(classification_count):
            response = requests.get("https://thispersondoesnotexist.com/")
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                self.process_and_upload_img(image)
            else:
                raise Exception(f"run] Failed to get face with error {response.status_code}")