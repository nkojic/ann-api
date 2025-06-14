import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Učitavanje SavedModel modela
MODEL_DIR = "model_tf"
model = tf.saved_model.load(MODEL_DIR)
predict_fn = model.signatures["serving_default"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['inputs']
        input_tensor = tf.convert_to_tensor([data], dtype=tf.float32)
        result = predict_fn(input_tensor)
        # Uzmi prvi rezultat iz dict-a
        prediction_value = list(result.values())[0].numpy()[0][0]
        return jsonify({'prediction': float(prediction_value)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
