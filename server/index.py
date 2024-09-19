from flask import Flask, jsonify, request
from ml_model.model import *
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)


def lower_size():
    model = MLModel(img_h=218,img_w=178, model_path="ml_model/models/Eyeglasses.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
    quantized_model = converter.convert()
    with open('quantized_model.tflite', 'wb') as f:
        f.write(quantized_model)

def test_reduced_model():
    interpreter = tf.lite.Interpreter(model_path=str("quantized_model.tflite"))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    print(input_index, output_index)

    directory = "./ml_model/data/Test"
    full_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    test = [image.load_img(img_path, target_size=(218, 178)) for img_path in full_paths]

    preds = []
    for img in test:
        inp = np.array(img)
        inp = inp.astype(np.float32)
        inp = np.expand_dims(inp, axis=0)
        interpreter.set_tensor(input_index, inp)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)
        preds.append(prediction.tolist())
    return preds

# lower_size()
print(test_reduced_model())

# model = MLModel(img_h=218,img_w=178, model_path="ml_model/models/Eyeglasses.keras")
@app.route('/incomes')
def get_incomes():
    # directory = "./ml_model/data/Test"
    # full_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))][0]
    # return model.predict(full_paths).tolist()
    return test_reduced_model()

@app.route('/')
def hello_world():
    return "hello-world"

app.run(host='0.0.0.0', port=5000)
