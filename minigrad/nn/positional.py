import numpy as np

class AIPPositionalEncoding:
    def __init__(self, d_model, maxlen=100):
        self.encoding = np.zeros((maxlen, d_model))
        position = np.arange(0, maxlen)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = np.sin(position * div_term)
        self.encoding[:, 1::2] = np.cos(position * div_term)

    def __call__(self, sequence_length):
        return self.encoding[:sequence_length, :]
