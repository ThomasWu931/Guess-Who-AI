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
    sample_size = 10
    sample_images = list(collection.aggregate([{"$sample": {"size": sample_size}}]))
    sample_images = [
        {
            k: str(v) if k == "_id" else v for k,v in image.items()
        }
        for image in sample_images
    ]
    return bson.BSON.encode({"data": sample_images})

@app.route("/get_question", methods=["POST"])
def get_question():
    data = request.get_json()
    image_ids = data["image_ids"]

    # Process questions
    questions = []
    for q in data["questions"]:
        if q["type"] == QuestionType.Guess.value:
            question = GuessQuestion(q["image_name"])
        elif q["type"] == QuestionType.Trait.value:
            question = TraitQuestion(Trait(q["trait"]))
        questions.append(question)
    
    # Process answers
    answers = [Answer(a) for a in data["answers"]]

    # Get best question
    s = MongoSolver(collection, image_ids, verbose=True)
    
    for q,a in zip(questions, answers):
        s.process_question_and_answer(q,a)

    question = s.get_best_question()
    if question.type == QuestionType.Guess:
        return {
            "type": question.type.value,
            "question": repr(question),
            "image_name": question.image_name
        }
    elif question.type == QuestionType.Trait:
        return {
            "type": question.type.value,
            "question": repr(question),
            "trait": question.trait.value
        }
    else:
        raise Exception(f"[get_question] Invalid type {question.type}")

# Run app
app.run(host='0.0.0.0', port=5678 )
