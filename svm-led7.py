from Manager import Manager
from ModelParameters import ModelParameters
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import math


def normalised_soft_max_decision_values(decision_values):
    decision_values = np.float64(decision_values)
    num_of_values = np.size(decision_values, axis=0)
    for indx in range(num_of_values):
        decision_value = decision_values[indx]
        min_val = np.min(decision_value)
        if min_val < 0:
            decision_value = np.add(decision_value, -1 * min_val)

        decision_value = np.divide(decision_value, np.sum(decision_value) + 1)
        decision_values[indx] = decision_value

    return np.multiply(decision_values, 100)

n_class = 10
input_file_path = "Train-LED.csv"
test_input_file_path = "Predict-LED.csv"
params = ModelParameters(number_of_hidden_layers=1,
                         number_of_input_neurons=28,
                         number_of_output_neurons=n_class,
                         number_of_hidden_layer_neurons=28,
                         epoch_size=10,
                         learning_rate=0.0015)

output_file_path = "all_cases_prediction_svm.csv"
project_name = "LED-7"
manager = Manager(project_name, params, input_file_path)
features, labels = manager.get_training_data(shuffle=False)
labels = labels.argmax(axis=1)

c = math.pow(2, n_class)
gamma = 1 / c

clf = svm.SVC(gamma=gamma, C=c)
clf.fit(features, labels)

test_manager = Manager(project_name, params, test_input_file_path)
test_features, _ = test_manager.get_training_data(shuffle=False)

decision_values = clf.decision_function(X=test_features)
decision_values = normalised_soft_max_decision_values(decision_values=decision_values)

content = "0,1,2,3,4,5,6,7,8,9\n"
for iter in range(len(decision_values)):
    row = decision_values[iter]
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
