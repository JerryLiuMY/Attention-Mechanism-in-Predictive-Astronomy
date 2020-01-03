import math
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error


class GP():
    def __init__(self):
        self.model = None

    def fit_model(self, t_list_train, mag_list_train, magerr_list_train):
        # Train gaussian process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        alpha = magerr_list_train ** 2
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
        self.model.fit(t_list_train.reshape(-1, 1), (mag_list_train-np.mean(mag_list_train)).reshape(-1, 1))  # MLE

        return self.model

    @staticmethod
    def match_list(model, t_list, mag_list_train):
        y_pred_num = int((t_list.max() - t_list.min()) / 0.2)
        t_pred = np.atleast_2d(np.linspace(t_list.min(), t_list.max(), y_pred_num)).T
        y_pred, y_pred_std = model.predict(t_pred, return_std=True)
        y_pred_std = np.expand_dims(y_pred_std, axis=1)
        y_pred = y_pred + np.mean(mag_list_train)

        y_pred_match = []
        y_pred_std_match = []
        for t in t_list:
            t_id = min(range(len(t_pred)), key=lambda i: abs(t_pred[i] - t))
            y_pred_match.append(y_pred[t_id])
            y_pred_std_match.append(y_pred_std[t_id])

        return t_pred, y_pred, y_pred_std, y_pred_match, y_pred_std_match

    def multi_step_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                              t_list_cross, mag_list_cross, magerr_list_cross):

        inter_train_match_return = self.match_list(self.model, t_list_train, mag_list_train)
        t_inter_train, y_inter_train, y_inter_train_std, y_inter_train_match, y_inter_train_sig_match = inter_train_match_return
        multi_train_loss = mean_squared_error(y_inter_train_match, mag_list_train)

        pred_cross_match_return = self.match_list(self.model, t_list_cross, mag_list_train)
        t_pred_cross, y_pred_cross, y_pred_cross_sig, y_pred_cross_match, y_pred_sig_match = pred_cross_match_return
        multi_cross_loss = mean_squared_error(y_pred_cross_match, mag_list_cross)

        multi_fig = self.plot_prediction(t_list_train, mag_list_train, magerr_list_train,
                                         t_list_cross, mag_list_cross, magerr_list_cross,
                                         t_inter_train, y_inter_train, y_inter_train_std,
                                         t_pred_cross, y_pred_cross, y_pred_cross_sig)

        return multi_train_loss, multi_cross_loss, multi_fig

    def plot_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                        t_list_cross, mag_list_cross, magerr_list_cross,
                        t_inter_train, y_inter_train, y_inter_train_sig,
                        t_pred_cross, y_pred_cross, y_pred_cross_sig):

        # print(np.shape(t_inter_train))
        # print(np.shape(y_inter_train_sig))
        # print(np.shape(y_inter_train))

        # Specify parameters
        max_time = t_list_cross.max()
        min_time = t_list_train.min()
        upper_limit = max_time
        lower_limit = min_time

        # Plot the function, the prediction and the 95% confidence interval based on the MSE
        multi_fig = plt.figure(figsize=(12, 8))
        plt.errorbar(t_list_train, mag_list_train, magerr_list_train, fmt='k.', markersize=10, label='Training')
        plt.errorbar(t_list_cross, mag_list_cross, magerr_list_cross, fmt='k.', markersize=10, label='Validation')
        plt.plot(t_inter_train, y_inter_train, 'g-')
        plt.fill(np.concatenate([t_inter_train, t_inter_train[::-1]]),
                 np.concatenate([y_inter_train - 1.9600 * y_inter_train_sig, (y_inter_train + 1.9600 * y_inter_train_sig)[::-1]]),
                 alpha=.5, fc='limegreen', ec='None')

        plt.plot(t_pred_cross, y_pred_cross, 'b-')
        plt.fill(np.concatenate([t_pred_cross, t_pred_cross[::-1]]),
                 np.concatenate([y_pred_cross - 1.9600 * y_pred_cross_sig, (y_pred_cross + 1.9600 * y_pred_cross_sig)[::-1]]),
                 alpha=.5, fc='DodgerBlue', ec='None')

        plt.xlim(lower_limit, upper_limit)
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('Simulated Path')
        plt.show()

        return multi_fig