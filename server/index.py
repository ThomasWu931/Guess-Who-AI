from flask import Flask, jsonify, request
from GWB.ml_model.model import *
app = Flask(__name__)

incomes = [
    { 'description': 'salary', 'amount': 5000 }
]
model = MLModel(img_h=218,img_w=178,weights_dir="ml_model/weights",feature="Eyeglasses",load_weights=True)
@app.route('/incomes')
def get_incomes():
    directory = "./ml_model/data/Test"
    full_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return model.predict(full_paths).tolist()

@app.route('/')
def hello_world():
    return "hello-world"

# @app.route('/incomes', methods=['POST'])
# def add_income():
#     breakpoint()
#     incomes.append(request.get_json())
#     return '', 204