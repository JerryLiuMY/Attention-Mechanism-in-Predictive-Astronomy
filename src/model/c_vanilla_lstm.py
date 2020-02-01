import numpy as np
from keras.layers import LSTM, Dense
from keras import Input, Model
from utils.phased_lstm import PhasedLSTM
from utils.tools import match_list, continuous_plot, discrete_plot
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

    def discrete_prediction(self, t_train, mag_train, magerr_train, t_pred_train, X_train,
                         t_cross, mag_cross, magerr_cross, t_pred_cross, X_cross,
                         mag_scaler, sm_type):

        if sm_type == 'single':
            y_pred_train, y_pred_var_train = self.discrete_single(X_train, mag_scaler)
            y_pred_cross, y_pred_var_cross = self.discrete_single(X_cross, mag_scaler)
        elif sm_type == 'multi':
            y_pred_train, y_pred_var_train = self.discrete_multi(X_train, mag_scaler)
            y_pred_cross, y_pred_var_cross = self.discrete_multi(X_cross, mag_scaler)

        else:
            raise Exception('Invalid sm_type')

        train_loss = mean_squared_error(y_pred_train[:, 0], mag_train[self.window_len + 1: -1])
        cross_loss = mean_squared_error(y_pred_cross[:, 0], mag_cross[self.window_len + 1: -1])

        sim_fig, res_fig = discrete_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_pred_var_train,
                                         t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_pred_var_cross)

        return sim_fig, res_fig, train_loss, cross_loss

    def continuous_prediction(self, t_train, mag_train, magerr_train, t_pred_train, X_train,
                              t_cross, mag_cross, magerr_cross, t_pred_cross, X_cross,
                              mag_scaler, sa_type):

        if sa_type == 'simulated':
            y_pred_train, y_pred_var_train = self.discrete_single(X_train, mag_scaler)
            y_pred_cross, y_pred_var_cross = self.discrete_single(X_cross, mag_scaler)
        elif sm_type == 'multi':
            y_pred_train, y_pred_var_train = self.discrete_multi(X_train, mag_scaler)
            y_pred_cross, y_pred_var_cross = self.discrete_multi(X_cross, mag_scaler)

        else:
            raise Exception('Invalid sm_type')

        train_loss = mean_squared_error(y_pred_train[:, 0], mag_train[self.window_len + 1: -1])
        cross_loss = mean_squared_error(y_pred_cross[:, 0], mag_cross[self.window_len + 1: -1])

        sim_fig, res_fig = discrete_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_pred_var_train,
                                         t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_pred_var_cross)

        return sim_fig, res_fig, train_loss, cross_loss

#     elif dc_type == 'continuous':
#     if sm_type == 'single':
#         pass
#
#     elif sm_type == 'multi':
#         pass
#
#     else:
#         raise Exception('Invalid sm_type')
#
# else:
# raise Exception('Invalid dc_type')


    def discrete_single(self, X, mag_scaler):
        scaled_y_pred = []
        X_step = X[[0]]  # shape = (1, window_len, 2)
        for i in range(np.shape(X)[0]):
            scaled_y_step = self.model.predict(X_step)  # shape = (1, 1)
            scaled_y_pred.append(scaled_y_step[0])  # scaled_y_step[0] -> shape = (num_feature)
            if i == np.shape(X)[0]-1:
                break

            scaled_t_step = X[i+1][[-1], -1:]  # shape = (1, 1)
            new_feature = np.concatenate((scaled_y_step, scaled_t_step), axis=1)  # shape = (1, 2)
            X_step = np.concatenate((X_step[0][1:, :], new_feature), axis=0)  # shape = (window_len, 2)
            X_step = np.expand_dims(X_step, axis=0)  # shape = (1, window_len, 2)

        scaled_y_pred = np.array(scaled_y_pred)  # shape = (num_step, 1)
        scaled_y_pred_var = np.zeros(len(scaled_y_pred))
        y_pred = mag_scaler.inverse_transform(scaled_y_pred)
        y_pred_var = mag_scaler.inverse_transform(scaled_y_pred + scaled_y_pred_var) - y_pred

        return y_pred, y_pred_var

    def discrete_multi(self, X, mag_scaler):
        scaled_y_pred = self.model.predict(X)
        scaled_y_pred_var = np.zeros(len(scaled_y_pred))
        y_pred = mag_scaler.inverse_transform(scaled_y_pred)
        y_pred_var = mag_scaler.inverse_transform(scaled_y_pred + scaled_y_pred_var) - y_pred

        return y_pred, y_pred_var

    def continuous_prediction(self, t, X, mag_scaler):
        scaled_y_pred = []
        X_step = X[[0]]  # shape = (1, window_len, 2)
        t_pred = np.linspace(t.min(), t.max(), num=int((t.max() - t.min()) / 0.2))
        y_pred_match, y_pred_var_match = match_list(t, t_pred, y_pred, y_pred_var)
        return t_pred, y_pred

