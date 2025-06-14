from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Učitavanje modela
model = tf.keras.models.load_model("ann_model.h5")


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['inputs']
    input_array = np.array([data])
    prediction = model.predict(input_array)[0][0]
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
