import tensorflow as tf
from ModelHandler import ModelHandler
from ModelParameters import ModelParameters


class Predictor(object):

    def __init__(self, project_name, prediction, params):
        self.PROJECT_NAME = project_name
        self.SAVER_OBJECT = tf.train.Saver()
        self.SESSION = None
        self.PREDICTION = prediction
        self.MODEL_HANDLER = ModelHandler(model_name=project_name)
        self.PARAMS = params

    def predict(self, features, activation_function=None):
        with tf.Session() as predict_session:
            self.SESSION = predict_session
            self.SESSION.run(tf.global_variables_initializer())
            model_restore_status = self.MODEL_HANDLER.restore_model(self.SAVER_OBJECT, self.SESSION)
            if not model_restore_status:
                print ("Oops.. couldn't find the stored the model at location %s" % self.MODEL_HANDLER.get_saved_path())
                return None

            if activation_function is not None:
                return self.SESSION.run([activation_function(self.PREDICTION)],
                             feed_dict={self.PARAMS.FEATURES: features})
            else:
                return self.SESSION.run([tf.nn.softmax(self.PREDICTION)],
                             feed_dict={self.PARAMS.FEATURES: features})
