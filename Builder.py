import numpy as np
import tensorflow as tf
from ModelParameters import ModelParameters


class Builder(object):

    def __init__(self, params):
        self.PARAMS = params
        self.PREDICTION = None
        self.COST = None
        self.COST_OPTIMIZER = None

    def build(self):
        self._initialize_building()
        return self.PREDICTION, self.COST, self.COST_OPTIMIZER

    def _initialize_building(self):
        self._initialize_weights_biases()
        self._initialize_layers()
        self._initialize_cost_optimizer()

    def _initialize_weights_biases(self):
        total_number_of_layers = self.PARAMS.NUMBER_OF_HIDDEN_LAYERS + 1
        for index in range(total_number_of_layers):
            if index == 0:
                self.PARAMS.WEIGHTS.append(tf.Variable(tf.truncated_normal([self.PARAMS.NUMBER_OF_INPUT_NEURONS,
                                                                self.PARAMS.NUMBER_OF_HIDDEN_LAYER_NEURONS],
                                                               stddev=0.1)))
                self.PARAMS.BIASES.append(tf.Variable(tf.constant(0.1,
                                                      shape=[self.PARAMS.NUMBER_OF_HIDDEN_LAYER_NEURONS])))
                continue

            if index == (total_number_of_layers - 1):
                self.PARAMS.WEIGHTS.append(tf.Variable(tf.truncated_normal([self.PARAMS.NUMBER_OF_HIDDEN_LAYER_NEURONS,
                                                                self.PARAMS.NUMBER_OF_OUTPUT_NEURONS],
                                                               stddev=0.1)))
                self.PARAMS.BIASES.append(tf.Variable(tf.constant(0.1,
                                                      shape=[self.PARAMS.NUMBER_OF_OUTPUT_NEURONS])))
                continue

            self.PARAMS.WEIGHTS.append(tf.Variable(tf.truncated_normal([self.PARAMS.NUMBER_OF_HIDDEN_LAYER_NEURONS,
                                                            self.PARAMS.NUMBER_OF_HIDDEN_LAYER_NEURONS],
                                                           stddev=0.1)))
            self.PARAMS.BIASES.append(tf.Variable(tf.constant(0.1,
                                                  shape=[self.PARAMS.NUMBER_OF_HIDDEN_LAYER_NEURONS])))

    def _initialize_layers(self):
        total_number_of_layers = self.PARAMS.NUMBER_OF_HIDDEN_LAYERS + 1
        for index in range(total_number_of_layers):
            if index == 0:
                self.PARAMS.LAYERS.append(
                    self.PARAMS.ACTIVATION_FUNCTION(tf.add(tf.matmul(self.PARAMS.FEATURES, self.PARAMS.WEIGHTS[index]), self.PARAMS.BIASES[index])))
                continue

            if index == (total_number_of_layers - 1):
                self.PREDICTION = tf.add(tf.matmul(self.PARAMS.LAYERS[index - 1], self.PARAMS.WEIGHTS[index]), self.PARAMS.BIASES[index])
                self.PARAMS.LAYERS.append(self.PREDICTION)
                continue

            self.PARAMS.LAYERS.append(
                self.PARAMS.ACTIVATION_FUNCTION(tf.add(tf.matmul(self.PARAMS.LAYERS[index - 1], self.PARAMS.WEIGHTS[index]), self.PARAMS.BIASES[index])))

    def _initialize_cost_optimizer(self):
        self.COST = self.PARAMS.TRAINING_PARAMETERS.REDUCE_MEAN(
            self.PARAMS.TRAINING_PARAMETERS.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS(
                logits=self.PREDICTION, labels=self.PARAMS.LABELS))

        self.COST_OPTIMIZER = self.PARAMS.OPTIMIZER.minimize(self.COST)

    def set_output_layer(self, output_layer):
        self.PREDICTION = output_layer
        self.PARAMS.set_output_layer(output_layer)