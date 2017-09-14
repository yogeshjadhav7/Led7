from ModelParameters import ModelParameters
from Builder import Builder
from Trainer import Trainer
from Predictor import Predictor
from DataHandler import DataHandler


class Manager(object):

    def __init__(self, project_name, params, input_file_path):
        self.PARAMS = params
        self.PROJECT_NAME = project_name
        self.INPUT_FILE_PATH = input_file_path
        self._initialize_manager()

    def _initialize_manager(self):
        self.BUILDER = None
        self.TRAINER = None
        self.PREDICTION = None
        self.COST = None
        self.COST_OPTIMIZER = None
        self.PREDICTOR = None
        self.DATA_HANDLER = DataHandler(self.INPUT_FILE_PATH, self.PARAMS.NUMBER_OF_INPUT_NEURONS, self.PARAMS.NUMBER_OF_OUTPUT_NEURONS)

    def get_training_data(self, shuffle=True):
        return self.DATA_HANDLER.extract_features_labels(shuffle)

    def build_simple_nn(self):
        self.BUILDER = Builder(self.PARAMS)
        self.PREDICTION, self.COST, self.COST_OPTIMIZER = self.BUILDER.build()

    def initialize_trainer(self):
        self.TRAINER = Trainer(self.PROJECT_NAME, self.PREDICTION, self.PARAMS,
                               self.COST_OPTIMIZER, self.COST)

    def train_simple_nn(self, features, labels):
        return self.TRAINER.train(features, labels)

    def initialize_predictor(self):
        self.PREDICTOR = Predictor(self.PROJECT_NAME, self.PREDICTION, self.PARAMS)

    def predict(self, features, activation_function=None):
        return self.PREDICTOR.predict(features, activation_function=activation_function)