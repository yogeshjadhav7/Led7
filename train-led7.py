from Manager import Manager
from ModelParameters import ModelParameters

input_file_path = "Train-LED.csv"
params = ModelParameters(number_of_hidden_layers=1,
                         number_of_input_neurons=28,
                         number_of_output_neurons=10,
                         number_of_hidden_layer_neurons=28,
                         epoch_size=10,
                         learning_rate=0.0001)

project_name = "LED-7"
manager = Manager(project_name, params, input_file_path)
manager.build_simple_nn()
manager.initialize_trainer()

cost_gloabal = 100
threshold = 1

while True:
    features, labels = manager.get_training_data()
    cost, before_accuracy, after_accuracy = manager.train_simple_nn(features, labels)
    if before_accuracy == 100:
        break

