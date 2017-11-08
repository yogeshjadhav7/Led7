from Manager import Manager
from ModelParameters import ModelParameters
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import math
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.tree import DecisionTreeClassifier


def treat_features(x):
    for indx in range(len(x)):
        row = x[indx]
        sum = np.sum(row) + 0.1
        row = np.multiply(row, 100 / sum)
        x[indx] = row

    return x


def normalised_softmax_decision_values(decision_values):
    decision_values = np.float64(decision_values)
    num_of_values = np.size(decision_values, axis=0)
    for indx in range(num_of_values):
        decision_value = decision_values[indx]
        min_val = np.min(decision_value)
        if min_val < 0:
            decision_value = np.add(decision_value, -1 * min_val)

        decision_value = np.divide(decision_value, np.sum(decision_value))
        decision_values[indx] = decision_value

    return np.multiply(decision_values, 100)


def find_optimal_dimensions_to_scale(x):
    stress_slopes = []
    n_features = np.size(x, axis=1)
    similarities = euclidean_distances(x.astype(np.float64))
    prev_x = 0
    prev_y = 0
    dimensions = SEED_DIMENSIONS
    while dimensions <= n_features:
        mds = MDS(n_components=dimensions)
        _ = mds.fit(similarities).embedding_
        stress = mds.stress_
        if prev_x == 0:
            prev_x = dimensions
            prev_y = stress
            dimensions = dimensions + 1
            continue

        diff_x = dimensions - prev_x
        diff_y = stress - prev_y
        stress_slopes.append(diff_y / diff_x)
        prev_x = dimensions
        prev_y = stress
        dimensions = dimensions + 1

    indx = np.argmin(stress_slopes)
    return SEED_DIMENSIONS + indx + 2


def mds_scaling(x, dimensions):
    mds = MDS(n_components=dimensions)
    similarities = euclidean_distances(x.astype(np.float64))
    return mds.fit(similarities).embedding_


SEED_DIMENSIONS = 2
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
features = features[:, [0,1,2,3,4,5,6]]
features = treat_features(features)
labels = labels.argmax(axis=1)

test_manager = Manager(project_name, params, test_input_file_path)
test_features, _ = test_manager.get_training_data(shuffle=False)
test_features = test_features[:, [0,1,2,3,4,5,6]]
test_features = treat_features(test_features)


dimensions = find_optimal_dimensions_to_scale(features)
print "Optimal Dimensions: " + str(dimensions)


#features = mds_scaling(features, dimensions)
#test_features = mds_scaling(test_features, dimensions)
#clf = svm.SVC(C=25, gamma=0.01, kernel='rbf')
#clf.fit(features, labels)
#print labels
#print clf.predict(features)
#decision_values = clf.decision_function(X=test_features)
#decision_values = normalised_softmax_decision_values(decision_values=decision_values)


clf = DecisionTreeClassifier(criterion='entropy', max_depth=dimensions)
clf = clf.fit(features, labels)
print labels
print clf.predict(features)
decision_values = clf.predict_proba(test_features)
decision_values = normalised_softmax_decision_values(decision_values=decision_values)


features, labels = test_manager.get_training_data(shuffle=False)
features = features[:, [0,1,2,3,4,5,6]]

list = []
content = "0,1,2,3,4,5,6,7,8,9\n"
for iter in range(len(decision_values)):
    row = decision_values[iter]
    print str(features[iter]) + " : " + str(np.argmax(row))
    list.append(np.argmax(row))
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
print list