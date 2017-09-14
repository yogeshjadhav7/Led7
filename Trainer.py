import tensorflow as tf
from ModelHandler import ModelHandler


class Trainer(object):

    def __init__(self, project_name, prediction, params, cost_optimizer, cost):
        self.PROJECT_NAME = project_name
        self.SAVER_OBJECT = tf.train.Saver()
        self.SESSION = None
        self.PREDICTION = prediction
        self.MODEL_HANDLER = ModelHandler(model_name=project_name)
        self.PARAMS = params
        self.COST_OPTIMIZER = cost_optimizer
        self.COST = cost

    def train(self, features, labels):
        with tf.Session() as session:
            self.SESSION = session
            self.SESSION.run(tf.global_variables_initializer())
            self.MODEL_HANDLER.restore_model(self.SAVER_OBJECT, self.SESSION)

            before_training_accuracy = self.__get_accuracy_percentage(features, labels)
            print ("Before training accuracy obtained: %s" % before_training_accuracy)

            epoch_loss = self.__enforce_learning(features, labels)

            after_training_accuracy = self.__get_accuracy_percentage(features, labels)
            print ("After training accuracy obtained: %s" % after_training_accuracy)

            self.MODEL_HANDLER.save_model(self.SAVER_OBJECT, self.SESSION)
            return epoch_loss, before_training_accuracy, after_training_accuracy

    def __get_accuracy_percentage(self, features, labels):
        model_predicted_labels = tf.argmax(self.PREDICTION, 1)
        data_labels = tf.argmax(labels, 1)
        correct = tf.equal(data_labels, model_predicted_labels)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_percentage = 100 * accuracy.eval(feed_dict={self.PARAMS.FEATURES: features,
                                                   self.PARAMS.LABELS: labels})

        return accuracy_percentage

    def __enforce_learning(self, features, labels):
        epoch_loss = 0
        for current_epoch in range(self.PARAMS.EPOCH_SIZE):
            _, current_epoch_loss = self.SESSION.run([self.COST_OPTIMIZER, self.COST],
                                                     feed_dict={self.PARAMS.FEATURES: features,
                                                                self.PARAMS.LABELS: labels})
            epoch_loss = epoch_loss + current_epoch_loss

        print ("Total Cost: %s" % epoch_loss)
        return epoch_loss


