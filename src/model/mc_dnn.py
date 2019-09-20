# Modified from 'https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py' by Yarin Gal
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.regularizers import l2
from keras import Input
from keras import Model
from keras.layers import Dropout, Dense
from scipy.special import logsumexp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

class MCDnn:

    def __init__(self, n_hidden, n_epochs=40, batch_size=128, tau=1.0, dropout=0.05, lengthscale=1e-2):
        self.dropout = dropout
        self.tau = tau
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lengthscale = lengthscale

        self.model = None
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def build_model(self, X_train, y_train):
        N = X_train.shape[0]
        reg = self.lengthscale ** 2 * (1 - self.dropout) / (2. * N * self.tau)

        inputs = Input(shape=(X_train.shape[1],))
        inter = Dropout(self.dropout)(inputs, training=True)
        inter = Dense(self.n_hidden[0], activation='relu', W_regularizer=l2(reg))(inter)
        for i in range(len(self.n_hidden) - 1):
            inter = Dropout(self.dropout)(inter, training=True)
            inter = Dense(self.n_hidden[i + 1], activation='relu', W_regularizer=l2(reg))(inter)
        inter = Dropout(self.dropout)(inter, training=True)
        outputs = Dense(y_train.shape[1], W_regularizer=l2(reg))(inter)
        self.model = Model(inputs, outputs)

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X_train, y_train, batch_size=self.batch_size, nb_epoch=self.n_epochs, verbose=0)

    def predict(self, X_test, y_test):
        # standard prediction
        standard_pred = self.model.predict(X_test, batch_size=500, verbose=1)

        # MC prediction
        T = 10000
        Yt_hat = np.array([self.model.predict(X_test, batch_size=500, verbose=0) for _ in range(T)])
        print(Yt_hat)
        MC_pred = np.mean(Yt_hat, 0)
        MC_std = np.std(Yt_hat, 0)

        # Test log-likelihood
        ll = logsumexp(-0.5*self.tau*(y_test[None]-Yt_hat)**2, 0) - np.log(T) - 0.5 * np.log(2*np.pi) + 0.5 * np.log(self.tau)
        test_ll = np.mean(ll)

        return standard_pred, MC_pred, MC_std, test_ll


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