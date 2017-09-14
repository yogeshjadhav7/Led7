import numpy as np
import pandas as pd


class DataHandler(object):

    def __init__(self, input_file_path, number_of_features, number_of_labels, skip_top_rows=1):
        self.INPUT_FILE_PATH = input_file_path
        self.NUMBER_OF_FEATURES = number_of_features
        self.NUMBER_OF_LABELS = number_of_labels
        self.SKIP_TOP_ROWS = skip_top_rows

    def extract_features_labels(self, shuffle=True):
        data_file = pd.read_csv(self.INPUT_FILE_PATH, skiprows=np.arange(self.SKIP_TOP_ROWS), header=None, skip_blank_lines=True)
        data = np.float64(data_file.values)
        if shuffle:
            np.random.shuffle(data)

        return np.hsplit(data, [self.NUMBER_OF_FEATURES])
