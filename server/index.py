from flask import Flask, jsonify, request
from ml_model.model import *

app = Flask(__name__)

incomes = [
    { 'description': 'salary', 'amount': 5000 }
]
# model = MLModel(img_h=218,img_w=178,weights_dir="./ml_model/weights",feature="Eyeglasses",load_weights=True)
# model.model.save("Eyeglasses.keras")
model = MLModel(img_h=218,img_w=178, model_path="ml_model/models/Eyeglasses.keras")
@app.route('/incomes')
def get_incomes():
    directory = "./ml_model/data/Test"
    full_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return model.predict(full_paths).tolist()

@app.route('/')
def hello_world():
    return "hello-world"

app.run(host='0.0.0.0', port=5000)
