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

    def build_model(self, X_train, y_train):
        # Build Model
        inputs = Input(shape=(X_train.shape[1], X_train.shape[2], ))
        if self.phased == 'phased':
            inter = PhasedLSTM(10, recurrent_dropout=self.dropout)(inputs, training=True)
        else:
            inter = LSTM(10, recurrent_dropout=self.dropout)(inputs, training=True)

        inter = Dense(10, activation='tanh')(inter)
        inter = Dropout(self.dropout)(inter, training=False)
        outputs = Dense(1, activation='linear')(inter)
        model = Model(inputs, outputs)
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

        return model

    def single_step_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                               t_list_cross, mag_list_cross, magerr_list_cross,
                               X_train, X_cross, X_test, mag_scaler, model):

        y_train_pred, y_train_std, single_train_loss = self.prediction(X_train, mag_list_train, mag_scaler, model)
        y_cross_pred, y_cross_std, single_cross_loss = self.prediction(X_cross, mag_list_cross, mag_scaler, model)

        single_fit_fig = self.plot_prediction(t_list_train, mag_list_train, magerr_list_train,
                                              t_list_cross, mag_list_cross, magerr_list_cross,
                                              y_train_pred, y_cross_pred)

        single_res_fig = self.plot_residual(t_list_train, mag_list_train, magerr_list_train,
                                            t_list_cross, mag_list_cross, magerr_list_cross,
                                            y_train_pred, y_cross_pred)

        return y_train_std, y_cross_std, single_train_loss, single_cross_loss, single_fit_fig, single_res_fig

    # ----------------------------------- Helper Function -----------------------------------
    def prediction(self, X, mag_list, mag_scaler, model):
        # Train Interpolation
        scaled_y_pred_ensemble = [model.predict(X, batch_size=self.batch_size, verbose=1) for _ in range(self.walkers)]
        y_pred_ensemble = [mag_scaler.inverse_transform(scaled_y_pred) for scaled_y_pred in scaled_y_pred_ensemble]
        y_pred = np.mean(y_pred_ensemble, axis=0)
        y_std = np.std(y_pred_ensemble, axis=0)
        loss = mean_squared_error(y_pred[:, 0], mag_list[self.window_len + 1: -1])

        return y_pred, y_std, loss

    def plot_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                        t_list_cross, mag_list_cross, magerr_list_cross,
                        y_inter, y_pred_cross):

        # Specify parameters
        max_time = t_list_cross.max()
        min_time = t_list_train.min()
        upper_limit = max_time
        lower_limit = min_time

        # Plot the function, the prediction and the 95% confidence interval based on the MSE
        fit_fig = plt.figure(figsize=(12, 8))
        plt.errorbar(t_list_train, mag_list_train, magerr_list_train, fmt='k.', markersize=10, label='Training')
        plt.errorbar(t_list_cross, mag_list_cross, magerr_list_cross, fmt='k.', markersize=10, label='Validation')
        plt.scatter(t_list_train[self.window_len+1: -1], y_inter, color='g')
        plt.scatter(t_list_cross[self.window_len+1: -1], y_pred_cross, color='b')
        plt.xlim(lower_limit, upper_limit)
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('Simulated Path')
        plt.show()

        return fit_fig

    def plot_residual(self, t_list_train, mag_list_train, magerr_list_train,
                      t_list_cross, mag_list_cross, magerr_list_cross,
                      y_inter, y_pred_cross):

        # Specify parameters
        max_time = t_list_cross.max()
        min_time = t_list_train.min()
        upper_limit = max_time
        lower_limit = min_time

        # Residual
        y_residual_inter = y_inter[:, 0] - mag_list_train[self.window_len+1: -1]
        y_residual_pred = y_pred_cross[:, 0] - mag_list_cross[self.window_len+1: -1]
        y_zeros_train = np.zeros(len(magerr_list_train))
        y_zeros_cross = np.zeros(len(magerr_list_cross))

        # Plot the function, the prediction and the 95% confidence interval based on the MSE
        res_fig = plt.figure(figsize=(12, 8))
        plt.errorbar(t_list_train, y_zeros_train, magerr_list_train,
                     fmt='k.', markersize=10, label='Training')
        plt.errorbar(t_list_cross, y_zeros_cross, magerr_list_cross,
                     fmt='k.', markersize=10, label='Validation')
        plt.scatter(t_list_train[self.window_len+1: -1], y_residual_inter, color='g')
        plt.scatter(t_list_cross[self.window_len+1: -1], y_residual_pred, color='b')
        plt.xlim(lower_limit, upper_limit)
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('Residual Plot')
        plt.show()

        return res_fig


if __name__ == '__main__':
    import json
    pass
