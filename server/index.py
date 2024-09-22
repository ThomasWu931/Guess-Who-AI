from flask import Flask, jsonify, request
from dotenv import load_dotenv
import pymongo
import os
from PIL import Image
from io import BytesIO

# Get collection
load_dotenv()
mongo_url = os.getenv('MONGO_URL')
db_name = os.getenv('DB_NAME')
collection_name = os.getenv("COLLECTION_NAME")

client = pymongo.MongoClient(mongo_url)
db = client[db_name]
collection = db[collection_name]

app = Flask(__name__)

@app.route('/images', methods=['GET'])
def get_images():
    # Sample 25 images from the db
    sample_size = 1
    sample = list(collection.aggregate([{"$sample": {"size": sample_size}}]))
    return [s["Eyeglasses"] for s in sample]

@app.route('/')
def hello_world():
    return "hello-world"

# Run app
app.run(host='0.0.0.0', port=5678 )
