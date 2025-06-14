import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import os
import zipfile

app = Flask(__name__)

# Putanja do zip fajla i gde će se raspakovati
ZIP_PATH = "/etc/secrets/model_tf.zip"
MODEL_DIR = "/tmp/model_tf"

# Raspakuj model ako već nije
if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# Učitaj model iz SavedModel foldera
model = tf.keras.models.load_model(MODEL_DIR)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['inputs']
    input_array = np.array([data])
    prediction = model.predict(input_array)[0][0]
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

