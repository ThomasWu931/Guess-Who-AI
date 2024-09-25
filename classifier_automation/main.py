import os
from dotenv import load_dotenv
from .ClassificationAutomator import ClassificationAutomator
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image

def main():
    load_dotenv()  # take environment variables from .env.

    # Get MongoDB connection URL from the .env file
    mongo_url = os.getenv('MONGO_URL')
    db_name = os.getenv('DB_NAME')
    collection_name = os.getenv("COLLECTION_NAME")

    # Get models
    models = {
        "Eyeglasses": load_model("ml_model\models\Eyeglasses.keras"), # Note that this import is relative to the working directory which initialized python code
        "Male": load_model("ml_model\models\Male.keras"),
        "Blond_hair": load_model("ml_model\models\Blond_hair.keras")
    }

    # Run
    automator = ClassificationAutomator(mongo_url, db_name, collection_name, models)
    automator.run(30)

    data = list(automator.collection.find())            
    # Loop through the cursor to print each document
    for document in data:
        image = Image.open(BytesIO(document["image"]))
        image.show()

main()

