import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from global_setting import DATA_FOLDER
from sklearn.metrics import mean_squared_error


def plot_series():
    plt.figure(figsize=(15, 15))
    i = 0
    for file in glob.glob(os.path.join(DATA_FOLDER, '*.csv')):
        i += 1
        with open(file) as handle:
            content = pd.read_csv(handle)
            mag_list = np.array(content['Mag'])
            mjd = np.array(content['MJD'])
            plt.subplot(5, 2, i)
            plt.scatter(mjd, mag_list)
            plt.xlabel('MJD')
            plt.ylabel('Mag')
            plt.title(file.partition(file)[0])

    plt.show()


def plot_series_short():
    plt.figure(figsize=(15, 15))
    i = 0
    for file in glob.glob(os.path.join(DATA_FOLDER, '*.csv')):
        i += 1
        with open(file) as handle:
            content = pd.read_csv(handle)
            mag_list = np.array(content['Mag'])[0:30]
            magerr_list = np.array(content['Magerr'])[0:30]
            mjd = np.array(content['MJD'])[0:30]
            plt.subplot(5, 2, i)
            plt.errorbar(mjd, mag_list, yerr=magerr_list, fmt='o')
            plt.xlabel('MJD')
            plt.ylabel('Mag')
            plt.title(file.partition(file)[0])

    plt.show()


def average_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_pred_var_train,
                 t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_pred_var_cross):

    average_fig = plt.figure(figsize=(12, 8))
    plt.errorbar(t_train, mag_train, yerr=magerr_train, fmt='k.')
    plt.errorbar(t_train, mag_train, yerr=magerr_cross, fmt='k.')

    # Interpolation
    plt.plot(t_pred_train, y_pred_train, color='green', ls='-')
    plt.fill_between(t_pred_train, y1=y_pred_train + np.sqrt(y_pred_var_train),
                     y2=y_pred_train - np.sqrt(y_pred_var_train), color='LimeGreen', alpha=0.5)

    # Prediction
    plt.plot(t_pred_cross, y_pred_cross, color='blue', ls='-')
    plt.fill_between(t_pred_cross, y1=y_pred_cross + np.sqrt(y_pred_var_cross),
                     y2=y_pred_cross - np.sqrt(y_pred_var_cross), color='DodgerBlue', alpha=0.5)

    # Decoration
    plt.xlim(t_train.min(), t_cross.max())
    plt.xlabel('Time[days]')
    plt.ylabel('Magnitude')
    plt.title('Simulated Average Paths')

    return average_fig


def sample_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train_n,
                t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross_n):

    sample_fig = plt.figure(figsize=(12, 8))
    plt.errorbar(t_train, mag_train, yerr=magerr_train, fmt='k.')
    plt.errorbar(t_cross, mag_cross, yerr=magerr_cross, fmt='k.')

    # Interpolation
    n_paths = np.shape(y_pred_train_n)[0]
    for i in range(n_paths):
        y_pred_train = y_pred_train_n[i]
        plt.plot(t_pred_train, y_pred_train, color='green', alpha=(1-i/float(n_paths)))

    # Prediction
    for i in range(n_paths):
        y_pred_cross = y_pred_cross_n[i]
        plt.plot(t_pred_cross, y_pred_cross, color='blue', alpha=(1-i/float(n_paths)))

    # Decoration
    plt.xlim(t_train.min(), t_cross.max())
    plt.xlabel('Time[days]')
    plt.ylabel('Magnitude')
    plt.title('Simulated Sample Paths')

    return sample_fig


def simulated_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train,
                   t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, type):
    scatter_plot = plt.figure(1, figsize=(12, 8))
    plt.errorbar(t_train, mag_train, magerr_train, fmt='k.', markersize=10, label='Training')
    plt.errorbar(t_cross, mag_cross, magerr_cross, fmt='k.', markersize=10, label='Validation')
    plt.scatter(t_pred_train, y_pred_train, color='g')
    plt.scatter(t_pred_cross, y_pred_cross, color='b')

    # Decoration
    plt.xlim(t_train.min(), t_cross.max())
    plt.xlabel('MJD')
    plt.ylabel('Mag')
    plt.legend(loc='upper left')
    plt.title('Simulated Plot') if type == 'Simulated' else plt.title('Residual Plot')
    plt.show()

    return simulated_plot


def residual_plot(t_train, mag_train, magerr_train, t_pred_train, residual_y_pred_train,
                  t_cross, mag_cross, magerr_cross, t_pred_cross, residual_y_pred_cross, type):
    residual_plot = plt.figure(1, figsize=(12, 8))
    zeros_train = np.zeros(len(magerr_train))
    zeros_cross = np.zeros(len(magerr_cross))
    plt.errorbar(t_train, zeros_train, magerr_train, fmt='k.', markersize=10, label='Training')
    plt.errorbar(t_cross, zeros_cross, magerr_cross, fmt='k.', markersize=10, label='Validation')
    plt.scatter(t_pred_train, residual_y_pred_train, color='g')
    plt.scatter(t_pred_cross, residual_y_pred_cross, color='b')

    plt.errorbar(t_train, mag_train, magerr_train, fmt='k.', markersize=10, label='Training')
    plt.errorbar(t_cross, mag_cross, magerr_cross, fmt='k.', markersize=10, label='Validation')

    return residual_plot


def match_list(t, t_pred, y_pred, y_pred_var):
    y_pred_match = []
    y_pred_var_match = []
    for date in t:
        t_id = min(range(len(t_pred)), key=lambda i: abs(t_pred[i] - date))
        y_pred_match.append(y_pred[t_id])
        y_pred_var_match.append(y_pred_var[t_id])

    return y_pred_match, y_pred_var_match


num = (t_train.max() - t_train.min()) / 0.2
t_inter = np.linspace(t_train.min(), t_train.max(), num=num)
