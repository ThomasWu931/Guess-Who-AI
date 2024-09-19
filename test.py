import tensorflow as tf
import numpy as np

# Load the TFLite model from the file
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")

# Allocate tensors (needed for setting input and output tensors)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check model input details
print(f"Input details: {input_details}")
# Ensure your input shape is correct (e.g., [1, img_h, img_w, num_channels])
input_shape = input_details[0]['shape']
print(f"Expected input shape: {input_shape}")

# Example input (random data) - ensure this matches the model's expected input
# Replace this with actual image input
input_data = np.random.random_sample(input_shape).astype(np.float32)  # Example input
# If the model expects normalized values (0 to 1), you may need to normalize your data here

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"Inference output shape: {output_data.shape}")
print("Inference output:", output_data)
