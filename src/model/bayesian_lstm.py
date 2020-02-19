import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from keras.models import Sequential
from utils.phased_lstm import PhasedLSTM
from sklearn.metrics import mean_squared_error
from keras import Input, Model
from keras.layers import Dropout
np.random.seed(1)


class BayesianLSTM:
    # The model is now trained individually for each sample, so we feed in the crts_id for now
    def __init__(self, window_len, hidden_dim, epochs, batch_size, phased, walkers, dropout):
        # Configuration
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.phased = phased
        self.walkers = walkers
        self.dropout = dropout



    # ----------------------------------- Helper Function -----------------------------------
    def prediction(self, X, mag_list, mag_scaler, model):
        # Train Interpolation
        scaled_y_pred_ensemble = [model.predict(X, batch_size=self.batch_size, verbose=1) for _ in range(self.walkers)]
        y_pred_ensemble = [mag_scaler.inverse_transform(scaled_y_pred) for scaled_y_pred in scaled_y_pred_ensemble]
        y_pred = np.mean(y_pred_ensemble, axis=0)
        y_std = np.std(y_pred_ensemble, axis=0)
        loss = mean_squared_error(y_pred[:, 0], mag_list[self.window_len + 1: -1])

        return y_pred, y_std, loss


if __name__ == '__main__':
    pass
