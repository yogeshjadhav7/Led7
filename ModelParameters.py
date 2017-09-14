import numpy as np
import tensorflow as tf


class ActivationParameters(object):

    def __init__(self):
        self.RELU = tf.nn.relu
        self.RELU_U6 = tf.nn.relu6
        self.SIGMOID = tf.nn.sigmoid
        self.TANH = tf.nn.tanh


class Optimizers(object):

    def __init__(self, learning_rate):
        self.ADAM_OPTIMIZER = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.GRADIENT_DESCENT_OPTIMIZER = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)


class TrainingParameters(object):

    def __init__(self):
        self.REDUCE_MEAN = tf.reduce_mean
        self.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS = tf.nn.softmax_cross_entropy_with_logits


class ModelParameters(object):

    def __init__(self, number_of_input_neurons=16, number_of_output_neurons=4,
                 number_of_hidden_layers=3, number_of_hidden_layer_neurons=8,
                 epoch_size=25, learning_rate=0.0015):
        self.NUMBER_OF_HIDDEN_LAYERS = number_of_hidden_layers
        self.NUMBER_OF_INPUT_NEURONS = number_of_input_neurons
        self.NUMBER_OF_OUTPUT_NEURONS = number_of_output_neurons
        self.NUMBER_OF_HIDDEN_LAYER_NEURONS = number_of_hidden_layer_neurons
        self.EPOCH_SIZE = epoch_size
        self.LEARNING_RATE = learning_rate
        self.__inititalize_parameters()

    def __inititalize_parameters(self):
        self.ACTIVATION_PARAMETERS = ActivationParameters()
        self.TRAINING_PARAMETERS = TrainingParameters()
        self.OPTIMIZERS = Optimizers(self.LEARNING_RATE)

        self.ACTIVATION_FUNCTION = self.ACTIVATION_PARAMETERS.RELU
        self.OPTIMIZER = self.OPTIMIZERS.ADAM_OPTIMIZER

        self.FEATURES = tf.placeholder('float', [None, self.NUMBER_OF_INPUT_NEURONS])
        self.LABELS = tf.placeholder('float', [None, self.NUMBER_OF_OUTPUT_NEURONS])
        self.WEIGHTS = list()
        self.BIASES = list()
        self.LAYERS = list()

    def set_output_layer(self, output_layer):
        self.LAYERS.pop()
        self.LAYERS.append(output_layer)


