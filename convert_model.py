import tensorflow as tf

# Učitaj već sačuvan .h5 model
model = tf.keras.models.load_model("ann_model.h5")

# Eksportuj u TensorFlow SavedModel format (folder)
model.export("model_tf")

print("Model je uspešno eksportovan kao TensorFlow SavedModel format.")
