import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Nema ZIP-a — direktno iz lokalne putanje
MODEL_DIR = "model_tf"
model = tf.keras.models.load_model(MODEL_DIR)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['inputs']
    input_array = np.array([data], dtype=np.float32)  # eksplicitno float32
    prediction = model(input_array).numpy()[0][0]
    return jsonify({'prediction': float(prediction)})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

loaded = tf.saved_model.load("model_tf")
print(list(loaded.signatures["serving_default"].structured_input_signature))
