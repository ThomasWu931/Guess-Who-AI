from flask import Flask, jsonify, request
from dotenv import load_dotenv
import pymongo
import os
from PIL import Image
from io import BytesIO
import bson
from ..solver.mongo_solver_adapter import *
from flask_cors import CORS

# Get collection
load_dotenv()
mongo_url = os.getenv('MONGO_URL')
db_name = os.getenv('DB_NAME')
collection_name = os.getenv("COLLECTION_NAME")

client = pymongo.MongoClient(mongo_url)
db = client[db_name]
collection = db[collection_name]

app = Flask(__name__)
CORS(app)  # Allow all origins (for development purposes) (#TODO: EDIT THIS WHEN DEPLOYING TO BE MORE RESTRICTIVE)

@app.route('/images', methods=['GET'])
def get_images():
    # Sample 25 images from the db
    sample_size = 1
    sample_images = list(collection.aggregate([{"$sample": {"size": sample_size}}]))
    sample_images = [
        {
            k: str(v) if k == "_id" else v for k,v in image
        }
        for image in sample_images
    ]
    return bson.BSON.encode(sample_images)

@app.route("/get_question", methods=["POST"])
def get_question():
    data = request.get_json()
    image_ids = data["image_ids"]
    questions = data["questions"]
    answers = data["answers"]
    s = MongoSolver(collection, image_ids, questions, answers)
    return s.get_best_question_adapter(image_ids)

# Run app
app.run(host='0.0.0.0', port=5678 )
