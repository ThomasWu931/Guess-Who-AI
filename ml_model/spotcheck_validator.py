import requests
import numpy as np
from keras.preprocessing import image
from io import BytesIO
from PIL import Image
from keras import Model
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

class SpotcheckValidator:
    def __init__(self, model_paths: list[str]) -> None:
        """Class used for checking the performance of finalized models against validation data"""
        self.models: list[Model] = [load_model(model_path) for model_path in model_paths]
        self.model_names: list[str] = [model_path for model_path in model_paths]
        _, self.h, self.w, _ = self.models[0].input.shape
    
    def get_validation_data(self, amt=20):
        """Fetches amt images from https://thispersondoesnotexist.com/ and returns them as numpy arrays."""
        data = []
        for _ in range(amt):
            try:
                # Fetch the image
                response = requests.get("https://thispersondoesnotexist.com/", timeout=5)

                img = Image.open(BytesIO(response.content))
                img = img.resize((self.w, self.h))
                
                # Convert to numpy array
                img_array = image.img_to_array(img)
                
                # Append to data list
                data.append(img_array)
            except Exception as e:
                print(f"Failed to fetch image: {e}")
        return np.array(data)

    def validate(self):
        """Validates the models against the validation data"""
        validation_data = self.get_validation_data()
        results = {}
        for model_name, model in zip(self.model_names, self.models):
            # Assuming models have a method predict
            predictions = model.predict(validation_data).tolist()
            results[model_name] = predictions

        for model_name in results:
            self.visualize_predictions(validation_data, results[model_name], model_name)
        return results
    
    def visualize_predictions(self, validation_data, predictions, model_name):
        """Visualizes the predictions with the corresponding images."""
        num_images = len(validation_data)
        plt.figure(figsize=(14, 8))
        plt.suptitle(f"Model: {model_name}", fontsize=16)  # y=1.02 pushes the title above the subplots

        for i in range(num_images):
            img_array = validation_data[i]
            img = Image.fromarray(img_array.astype(np.uint8))  # Convert back to image

            plt.subplot(5, 4, i + 1)
            plt.imshow(img)
            plt.axis('off')

            # Display the prediction as the title (converting the prediction to a human-readable string)
            pred_text = f"{model_name} Prediction: {round(predictions[i][0], 2)}"
            plt.title(pred_text, fontsize=10)
        
        plt.tight_layout()
        plt.show()


root_model_dir = "./models"
checker = SpotcheckValidator([
    f"{root_model_dir}/{file}" for file in os.listdir(root_model_dir)
])

print(checker.validate())
