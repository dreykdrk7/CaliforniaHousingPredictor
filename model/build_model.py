import tensorflow as tf
from tensorflow.keras import layers

def build_model(input_shape, layers_config, activation, optimizer, loss, metrics):
    model = tf.keras.Sequential()
    for units in layers_config:
        model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
