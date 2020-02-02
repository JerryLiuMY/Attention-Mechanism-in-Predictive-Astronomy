import numpy as np
from keras.layers import LSTM, Dense
from keras import Input, Model
from utils.phased_lstm import PhasedLSTM
from utils.tools import match_list, continuous_plot, discrete_plot, WINDOW_LEN
from global_setting import N_WALKERS
from sklearn.metrics import mean_squared_error
np.random.seed(1)

# standard / drop_out / mixture model
# phased / no phased -- training
# discrete / continuous -- prediction
# single / multiple -- prediction
# discrete: simulated + residual or continuous: sample + average  -- plot


def mc_std(func):
    def wrapper(n_walkers, *args, **kwargs):
        t_pred, y_pred = func(*args, **kwargs)
        y_pred_n = []
        for i in range(n_walkers):
            t_pred, y_pred = func(*args, **kwargs)
            y_pred_n.append(y_pred)
        y_pred_n = np.array(y_pred_n)
        y_pred = np.mean(y_pred_n, axis=0)
        y_std = np.std(y_pred_n, axis=0)
        return t_pred, y_pred, y_std, y_pred_n
    return wrapper


class VanillaLSTM:
    # The model is now trained individually for each sample, so we feed in the crts_id for now
    def __init__(self, hidden_dim, epochs, batch_size, phased, n_walkers, **kwargs):
        # Configuration
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.phased = phased
        self.n_walkers = n_walkers
        self.model = None

    def fit_model(self, X_train, y_train):
        inputs = Input(shape=(WINDOW_LEN, 2, ))
        if self.phased == 'phased':
            inter = PhasedLSTM(self.hidden_dim)(inputs, training=True)
        else:
            inter = LSTM(self.hidden_dim)(inputs, training=True)

        inter = Dense(self.hidden_dim, activation='tanh')(inter)
        outputs = Dense(1, activation='linear')(inter)

        self.model = Model(inputs, outputs)
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

    def run_model(self, X, **kwargs):
        scaled_y_pred = self.model.predict(X)
        return scaled_y_pred

    def prediction(self, t_train, mag_train, magerr_train, X_train,
                         t_cross, mag_cross, magerr_cross, X_cross,
                         mag_scaler, delta_t_scaler, dc_type, sm_type):

        if dc_type == 'discrete':
            discrete_train = self.discrete(self.n_walkers, t_train, X_train, mag_scaler, delta_t_scaler, sm_type)
            discrete_cross = self.discrete(self.n_walkers, t_cross, X_cross, mag_scaler, delta_t_scaler, sm_type)
            t_pred_cross, y_pred_cross, y_std_cross, _ = discrete_cross
            t_pred_train, y_pred_train, y_std_train, _ = discrete_train

            train_loss = mean_squared_error(y_pred_train, mag_train[WINDOW_LEN:])
            cross_loss = mean_squared_error(y_pred_cross, mag_cross[WINDOW_LEN:])

            fig = discrete_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_std_train,
                                t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_std_cross)

        elif dc_type == 'continuous':
            continuous_train = self.continuous(self.n_walkers, t_train, X_train, mag_scaler, delta_t_scaler, sm_type)
            continuous_cross = self.continuous(self.n_walkers, t_cross, X_cross, mag_scaler, delta_t_scaler, sm_type)
            t_pred_train, y_pred_train, y_std_train, y_pred_train_n = continuous_train
            t_pred_cross, y_pred_cross, y_std_cross, y_pred_cross_n = continuous_cross

            _, y_pred_train_match, y_std_train_match = match_list(t_train, t_pred_train, y_pred_train, y_std_train)
            _, y_pred_cross_match, y_std_cross_match = match_list(t_train, t_pred_train, y_pred_train, y_std_train)
            train_loss = mean_squared_error(y_pred_train_match, mag_train[WINDOW_LEN:])
            cross_loss = mean_squared_error(y_pred_cross_match, mag_cross[WINDOW_LEN:])

            fig = continuous_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_std_train, y_pred_train_n,
                                  t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_std_cross, y_pred_cross_n)

        else:
            raise Exception('Invalid dc_type')

        return fig, train_loss, cross_loss

    @mc_std
    def discrete(self, t, X, mag_scaler, delta_t_scaler, sm_type):

        if sm_type == 'multiple':
            t_pred = t[WINDOW_LEN-1:]
            y_pred = mag_scaler.inverse_transform(self.model.predict(X))
            t_pred = t_pred[1:]

        elif sm_type == 'single':
            t_pred = t[WINDOW_LEN-1:]
            y_pred = self.recursive(t_pred, X, mag_scaler, delta_t_scaler)
            t_pred = t_pred[1:]

        else:
            raise Exception('Invalid sm_type')

        return t_pred, y_pred

    @mc_std
    def continuous(self, t, X, mag_scaler, delta_t_scaler, sm_type):

        if sm_type == 'multiple':
            t_pred = np.linspace(t[WINDOW_LEN-1], t[len(t)-1], num=int((t[-1] - t[WINDOW_LEN+1]) / 0.2))
            y_pred = self.recursive(t_pred, X, mag_scaler, delta_t_scaler)
            t_pred = t_pred[1:]

        elif sm_type == 'single':
            t_pred = np.array([])
            y_pred = np.array([])
            for i in range(WINDOW_LEN-1, len(t)-1):
                X_i = X[[i-(WINDOW_LEN-1)]]
                t_pred_i = np.linspace(t[i], t[i+1], num=int((t[i+1] - t[i]) / 0.2))
                y_pred_i = self.recursive(t_pred_i, X_i, mag_scaler, delta_t_scaler)
                t_pred_i = t_pred_i[1:]

                t_pred = np.concatenate([t_pred, t_pred_i])
                y_pred = np.concatenate([y_pred, y_pred_i])

        else:
            raise Exception('Invalid sm_type')

        return t_pred, y_pred

    def recursive(self, t_pred, X, mag_scaler, delta_t_scaler):
        # shape(X) = (num_train_data - window_len, window_len, 2)
        # shape(scaled_delta_t_pred) = (num_train_data - window_len, 1)
        scaled_delta_t_pred = delta_t_scaler.transform(np.diff(t_pred).reshape(-1, 1))
        scaled_y_pred = []

        # Initialize --> change time interval
        initial_scaled_y_step = X[0, -1:, 0].reshape(1, 1)  # shape = (1, 1)
        initial_scaled_delta_t_step = scaled_delta_t_pred[0].reshape(1, 1)  # shape = (1, 1)
        initial_feature = np.concatenate((initial_scaled_y_step, initial_scaled_delta_t_step), axis=1)  # shape = (1, 2)
        X_step = np.concatenate((X[0, :-1, :], initial_feature), axis=0)  # shape = (window_len, 2)
        X_step = np.expand_dims(X_step, axis=0)  # shape = (1, window_len, 2)
        scaled_y_step = self.run_model(X_step)  # shape = (1, 1)
        scaled_y_pred.append(scaled_y_step[0])

        for i in range(1, len(scaled_delta_t_pred)):
            scaled_delta_t_step = scaled_delta_t_pred[i].reshape(1, 1)  # shape = (1, 1)
            new_feature = np.concatenate((scaled_y_step, scaled_delta_t_step), axis=1)  # shape = (1, 2)
            X_step = np.concatenate((X_step[0, 1:, :], new_feature), axis=0)  # shape = (window_len, 2)
            X_step = np.expand_dims(X_step, axis=0)  # shape = (1, window_len, 2)
            scaled_y_step = self.run_model(X_step)  # shape = (1, 1)
            scaled_y_pred.append(scaled_y_step[0])

        y_pred = mag_scaler.inverse_transform(np.array(scaled_y_pred))  # shape = (num_step, 1)
        y_pred = y_pred.reshape(-1)

        return y_pred
