import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Putanja do SavedModel foldera (mora biti u repozitorijumu)
MODEL_DIR = "model_tf"

# Učitaj model iz SavedModel formata
loaded_model = tf.saved_model.load(MODEL_DIR)
inference_func = loaded_model.signatures["serving_default"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['inputs']  # očekuje listu dužine 6
        input_array = tf.constant([data], dtype=tf.float32)
        result = inference_func(input_layer=input_array)
        prediction = result["dense_3"].numpy()[0][0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
