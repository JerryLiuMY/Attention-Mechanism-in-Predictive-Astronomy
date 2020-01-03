# Modified from 'https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py' by Yarin Gal
from sklearn.preprocessing import StandardScaler
from keras.regularizers import l2
from keras import Input, Model
from keras.layers import Dropout, Dense
from scipy.special import logsumexp
import numpy as np
import json

def build_dataset(N, noise_std=0.25, is_test=True):
    x = np.random.rand(N, 1)

    if is_test:
        x = 8 * x - 4
    else:
        x = 6 * x - 3
        x[x < 0] -= 1
        x[x > 0] += 1

    # data with noise
    y = 0.1 * x ** 3 + np.random.normal(0, noise_std, size=(N, 1))

    return x, y

class DNN:

    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.load_config()

    def load_config(self):
        self.model_config = json.load(open('./config/model_config.json'))
        self.n_layers = self.model_config['dnn']['n_layers']
        self.n_units = self.model_config['dnn']['n_units']
        self.dropout = self.model_config['dnn']['dropout']
        self.tau = self.model_config['dnn']['tau']
        self.epochs = self.model_config['dnn']['epochs']
        self.walkers = self.model_config['dnn']['walkers']
        self.batch_size = self.model_config['dnn']['batch_size']
        self.lengthscale = self.model_config['dnn']['lengthscale']

        # self.tau = [0.25, 0.5, 0.75]
        # self.dropout = [0.005, 0.01, 0.05, 0.1]

    def build_dnn_model(self, X_train, y_train):
        N = X_train.shape[0]
        reg = self.lengthscale ** 2 * (1 - self.dropout) / (2. * N * self.tau)

        inputs = Input(shape=(X_train.shape[1],))
        inter = Dense(self.n_units, activation='relu', kernel_regularizer=l2(reg))(inputs)
        for _ in range(self.n_layers - 1):
            inter = Dense(self.n_units, activation='relu', kernel_regularizer=l2(reg))(inter)
        outputs = Dense(y_train.shape[1], kernel_regularizer=l2(reg))(inter)
        model = Model(inputs, outputs)

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1)

        return model

    def build_mc_dnn_model(self, X_train, y_train):
        N = X_train.shape[0]
        reg = self.lengthscale ** 2 * (1 - self.dropout) / (2. * N * self.tau)

        inputs = Input(shape=(X_train.shape[1],))
        inter = Dropout(self.dropout)(inputs, training=True)
        inter = Dense(self.n_units, activation='relu', kernel_regularizer=l2(reg))(inter)
        for i in range(self.n_layers - 1):
            inter = Dropout(self.dropout)(inter, training=True)
            inter = Dense(self.n_units, activation='relu', kernel_regularizer=l2(reg))(inter)
        inter = Dropout(self.dropout)(inter, training=True)
        outputs = Dense(y_train.shape[1], kernel_regularizer=l2(reg))(inter)
        model = Model(inputs, outputs)

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1)

        return model

    def predict(self, X_test, model):
        # standard prediction
        standard_pred = model.predict(X_test, batch_size=self.batch_size, verbose=1)

        # MC prediction
        Yt_hat = np.array([model.predict(X_test, batch_size=self.batch_size, verbose=0) for _ in range(self.walkers)])
        MC_pred = np.mean(Yt_hat, 0)
        MC_std = np.std(Yt_hat, 0)

        # Test log-likelihood
        # ll = logsumexp(-0.5*self.tau*(y_test[None]-Yt_hat)**2, 0) - np.log(self.walkers) - 0.5 * np.log(2*np.pi) + 0.5 * np.log(self.tau)
        # test_ll = np.mean(ll)

        return standard_pred, MC_pred, MC_std


if __name__ == 'main':
    # Raw Data
    x_train_val, y_train_val = build_dataset(200, noise_std=0.25, is_test=False)
    x_test, y_test = build_dataset(80, noise_std=0.25, is_test=True)

    # Normalized Data
    num_training_examples = int(0.8 * np.shape(x_train_val)[0])
    x_train = x_train_val[0:num_training_examples, :]
    y_train = y_train_val[0:num_training_examples]
    x_validation = x_train_val[num_training_examples:, :]
    y_validation = y_train_val[num_training_examples:]
