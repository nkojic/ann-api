import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import os
import zipfile

app = Flask(__name__)

# Putanja do zip fajla i gde će se raspakovati
ZIP_PATH = "/etc/secrets/model_tf_v2.zip"
MODEL_DIR = "/tmp/model_tf"

# Raspakuj model ako već nije
if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# Učitaj SavedModel model
loaded_model = tf.saved_model.load(MODEL_DIR)
predict_fn = loaded_model.signatures["serving_default"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['inputs']  # očekuje se lista od 6 float brojeva
    input_array = np.array([data], dtype=np.float32)  # Oblik (1, 6)

    # Napravi dict sa imenom input tenzora
    input_dict = {'input_layer': tf.convert_to_tensor(input_array)}

    # Poziv modela
    result = predict_fn(**input_dict)
    prediction = result['dense_3'].numpy()[0][0]  # dense_3 je naziv izlaza

    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
