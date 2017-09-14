from Manager import Manager
from ModelParameters import ModelParameters
import numpy as np
import tensorflow as tf

input_file_path = "Predict-LED.csv"
params = ModelParameters(number_of_hidden_layers=1,
                         number_of_input_neurons=7,
                         number_of_output_neurons=10,
                         number_of_hidden_layer_neurons=8)

project_name = "LED-7"
manager = Manager(project_name, params, input_file_path)
features, labels = manager.get_training_data(shuffle=False)
manager.build_simple_nn()
manager.initialize_predictor()
predicted_labels = manager.predict(features=features, activation_function=tf.nn.softmax).pop()
predicted_labels = np.float64(predicted_labels)
prediction = predicted_labels.argmax(axis=1)
print prediction
print predicted_labels