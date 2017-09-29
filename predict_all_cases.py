from Manager import Manager
from ModelParameters import ModelParameters
import numpy as np
import tensorflow as tf

input_file_path = "all_cases.csv"
output_file_path = "all_cases_prediction.csv"
params = ModelParameters(number_of_hidden_layers=1,
                         number_of_input_neurons=28,
                         number_of_output_neurons=10,
                         number_of_hidden_layer_neurons=28)

project_name = "LED-7"
manager = Manager(project_name, params, input_file_path)
features, labels = manager.get_training_data(shuffle=False)
manager.build_simple_nn()
manager.initialize_predictor()
predicted_labels = manager.predict(features=features, activation_function=tf.nn.softmax).pop()
predicted_labels = np.multiply(predicted_labels, 100)
content = "0,1,2,3,4,5,6,7,8,9\n"

for iter in range(len(predicted_labels)):
    row = predicted_labels[iter]
    content_row = ""
    for iter_in in range(len(row)):
        val = "{0:.2f}".format(round(row[iter_in], 2))
        if content_row == "":
            content_row = content_row + val
        else:
            content_row = content_row + "," + val

    content_row = content_row + "\n"
    content = content + content_row

f = open(output_file_path, 'w')
f.write(content)
f.close()

