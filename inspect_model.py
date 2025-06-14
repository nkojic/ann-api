import tensorflow as tf

# Učitaj model iz foldera
loaded = tf.saved_model.load("model_tf")

# Prikaži strukturu ulaza (input signature)
print(list(loaded.signatures["serving_default"].structured_input_signature))
