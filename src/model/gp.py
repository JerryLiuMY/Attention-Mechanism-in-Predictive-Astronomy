import math
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from utils.data_loader import data_loader
from operator import itemgetter
import json
data_config = json.load(open('config/data_config.json'))
model_config = json.load(open('config/model_config.json'))
np.random.seed(1)


class GP():
    def __init__(self, crts_id):
        self.crts_id = crts_id

    @staticmethod
    def partition(length):
        train_percent = data_config["partition"]["train"]
        cross_percent = data_config["partition"]["cross"]
        test_percent = data_config["partition"]["test"]
        p1 = math.floor(length * train_percent)
        p2 = math.floor(length * (train_percent + cross_percent))

        index_list = list(range(length))
        shuffled_list = random.sample(index_list, len(index_list))
        index_train = shuffled_list[0:p1]
        index_cross = shuffled_list[p1:p2]
        index_test = shuffled_list[p2:]
        return index_train, index_cross, index_test

    @staticmethod
    def mse(y, y_pred):
        mse = np.mean((y - y_pred) ** 2)
        return mse

    def fit_gaussian(self):
        # Partition data
        mag_list, magerr_list, mjd_list = data_loader(self.crts_id)
        index_train, index_cross, index_test = self.partition(len(mag_list))

        mag_list_train = mag_list[index_train]
        mag_list_cross = mag_list[index_cross]
        mag_list_test = mag_list[index_test]
        magerr_list_train = magerr_list[index_train]
        magerr_list_cross = magerr_list[index_cross]
        magerr_list_test = magerr_list[index_test]
        mjd_list_train = mjd_list[index_train].reshape(-1, 1)
        mjd_list_cross = mjd_list[index_cross].reshape(-1, 1)
        mjd_list_test = mjd_list[index_test].reshape(-1, 1)

        # Train gaussian process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        alpha = magerr_list_train ** 2
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
        gp.fit(mjd_list_train, mag_list_train)  # MLE

        # Make predictions
        max_time = max(mjd_list)
        min_time = min(mjd_list)
        upper_limit = max_time + 0.2 * (max_time - min_time)
        lower_limit = min_time - 0.2 * (max_time - min_time)
        upper_limit = max_time
        lower_limit = min_time
        num = int(upper_limit - lower_limit)
        x = np.atleast_2d(np.linspace(lower_limit, upper_limit, num)).T
        y_pred, sigma = gp.predict(x, return_std=True)

        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        plt.figure(figsize=(12, 8))
        # plt.plot(x, self.f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
        plt.errorbar(mjd_list_train, mag_list_train, magerr_list_train, fmt='r.', markersize=10, label='Training')
        plt.errorbar(mjd_list_cross, mag_list_cross, magerr_list_cross, fmt='b.', markersize=10, label='Validation')
        plt.errorbar(mjd_list_test, mag_list_test, magerr_list_test, fmt='g.', markersize=10, label='Test')
        plt.plot(x, y_pred, 'b-', label='Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('crts_id: ' + str(self.crts_id))
        plt.show()

    def fit_gaussian_short(self):
        # Partition data
        mag_list, magerr_list, mjd_list = data_loader(self.crts_id)
        index_train, index_cross, index_test = self.partition(len(mag_list))

        mag_list_train = mag_list[index_train][0:40]
        mag_list_cross = mag_list[index_cross][0:40]
        mag_list_test = mag_list[index_test][0:40]
        magerr_list_train = magerr_list[index_train][0:40]
        magerr_list_cross = magerr_list[index_cross][0:40]
        magerr_list_test = magerr_list[index_test][0:40]
        mjd_list_train = mjd_list[index_train][0:40].reshape(-1, 1)
        mjd_list_cross = mjd_list[index_cross][0:40].reshape(-1, 1)
        mjd_list_test = mjd_list[index_test][0:40].reshape(-1, 1)

        # Train gaussian process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        alpha = magerr_list_train ** 2
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
        gp.fit(mjd_list_train, mag_list_train)  # MLE

        # Make predictions
        max_time = max(mjd_list)
        min_time = min(mjd_list)
        upper_limit = max_time + 0.2 * (max_time - min_time)
        lower_limit = min_time - 0.2 * (max_time - min_time)
        upper_limit = max_time
        lower_limit = min_time
        num = int(upper_limit - lower_limit)
        x = np.atleast_2d(np.linspace(lower_limit, upper_limit, num)).T
        y_pred, sigma = gp.predict(x, return_std=True)

        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        plt.figure(figsize=(12, 8))
        # plt.plot(x, self.f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
        plt.errorbar(mjd_list_train, mag_list_train, magerr_list_train, fmt='r.', markersize=10, label='Training')
        plt.errorbar(mjd_list_cross, mag_list_cross, magerr_list_cross, fmt='b.', markersize=10, label='Validation')
        plt.errorbar(mjd_list_test, mag_list_test, magerr_list_test, fmt='g.', markersize=10, label='Test')
        plt.plot(x, y_pred, 'b-', label='Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('crts_id: ' + str(self.crts_id))
        plt.show()