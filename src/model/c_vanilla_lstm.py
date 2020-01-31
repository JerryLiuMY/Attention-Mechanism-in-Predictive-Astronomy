import numpy as np
from keras.layers import LSTM, Dense
from keras import Input, Model
from utils.phased_lstm import PhasedLSTM
from utils.tools import average_plot, sample_plot, match_list
from sklearn.metrics import mean_squared_error
np.random.seed(1)


class VanillaLSTM:
    # The model is now trained individually for each sample, so we feed in the crts_id for now
    def __init__(self, window_len, hidden_dim, epochs, batch_size, phased, **kwargs):
        # Configuration
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.phased = phased
        self.model = None

    def build_model(self, X_train, y_train):
        inputs = Input(shape=(self.window_len, 2, ))
        if self.phased == 'phased':
            inter = PhasedLSTM(self.hidden_dim)(inputs, training=True)
        else:
            inter = LSTM(self.hidden_dim)(inputs, training=True)

        inter = Dense(self.hidden_dim, activation='tanh')(inter)
        outputs = Dense(1, activation='linear')(inter)

        self.model = Model(inputs, outputs)
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

    def continuous_prediction(self, t, mag, n):
        y_pred_num = int((t.max() - t.min()) / 0.2)
        t_pred = np.linspace(t.min(), t.max(), num=y_pred_num)
        y_pred, y_pred_var = self.model.predict(t_pred)

        y_pred_n = []
        for i in range(n):
            y_pred, y_pred_var = self.model.predict(t_pred)
            y_pred_n.append(y_pred)

        y_pred_match, y_pred_var_match = match_list(t, t_pred, y_pred, y_pred_var)
        loss = mean_squared_error(mag, y_pred_match)

        return t_pred, y_pred, y_pred_n, y_pred_var, loss

    def prediction(self, t_train, mag_train, magerr_train, X_train,
                   t_cross, mag_cross, magerr_cross, X_cross,
                   mag_scaler, sm, dc):

        if dc == 'discrete' and sm == 'single':
            single_y_pred_train = mag_scaler.inverse_transform(self.model.predict(X_train))
            single_train_loss = mean_squared_error(single_y_pred_train[:, 0], mag_train[self.window_len + 1: -1])
            single_y_pred_cross = mag_scaler.inverse_transform(self.model.predict(X_cross))
            single_cross_loss = mean_squared_error(single_y_pred_cross[:, 0], mag_cross[self.window_len + 1: -1])

            discrete_single_fit_fig = self.plot_prediction(t_train, mag_train, magerr_train, single_y_pred_train,
                                                           t_cross, mag_cross, magerr_cross, single_y_pred_cross)

            discrete_single_res_fig = self.plot_residual(t_train, mag_train, magerr_train, single_y_pred_train,
                                                         t_cross, mag_cross, magerr_cross, single_y_pred_cross)

        if dc == 'discrete' and sm == 'multi':
            multi_y_pred_train = mag_scaler.inverse_transform(self.one_step(X_train))
            multi_train_loss = mean_squared_error(multi_y_pred_train[:, 0], mag_train[self.window_len + 1: -1])
            multi_y_pred_cross = mag_scaler.inverse_transform(self.one_step(X_cross))
            multi_cross_loss = mean_squared_error(multi_y_pred_cross[:, 0], mag_cross[self.window_len + 1: -1])

            discrete_multi_fit_fig = self.plot_prediction(t_train, mag_train, magerr_train, multi_y_pred_train,
                                                          t_cross, mag_cross, magerr_cross, multi_y_pred_cross)

            discrete_multi_res_fig = self.plot_residual(t_train, mag_train, magerr_train, multi_y_pred_train,
                                                        t_cross, mag_cross, magerr_cross, multi_y_pred_cross)

        return single_train_loss, single_cross_loss, single_fit_fig, single_res_fig

    def one_step(self, X):
        # Cross Prediction
        scaled_y = []
        X_step = X[[0]]  # shape = (1, window_len, num_feature)
        print(np.shape(X))
        for i in range(np.shape(X)[0]):
            scaled_y_step = self.model.predict(X_step)  # shape = (1, num_feature)
            scaled_y.append(scaled_y_step[0])  # scaled_y_step[0] -> shape = (num_feature)
            if i == np.shape(X)[0]-1:
                break

            scaled_t_step = X[i+1][[-1], -1:]  # shape = (1, 1)
            new_feature = np.concatenate((scaled_y_step, scaled_t_step), axis=1)  # shape = (1, num_feature + 1)
            X_step = np.concatenate((X_step[0][1:, :], new_feature), axis=0)  # shape = (window_len, num_feature + 1)
            X_step = np.expand_dims(X_step, axis=0)  # shape = (1, window_len, num_feature + 1)

        scaled_y = np.array(scaled_y)  # shape = (num_step, num_feature)

        return scaled_y
