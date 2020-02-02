import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from global_setting import DATA_FOLDER, WINDOW_LEN


def discrete_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_std_train,
                  t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_std_cross):

    dis_fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))
    for ax in axes.ravel():
        ax.set_xlim(t_train.min(), t_cross.max())
        ax.set_xlabel('MJD')
        ax.set_ylabel('Mag')
        plt.legend()

    plt.subplot(1, 2, 1)
    plt.errorbar(t_train, mag_train, magerr_train, fmt='k.', markersize=10)
    plt.errorbar(t_cross, mag_cross, magerr_cross, fmt='k.', markersize=10)
    plt.errorbar(t_pred_train, y_pred_train, y_std_train, fmt='g-', label='Training')
    plt.errorbar(t_pred_cross, y_pred_cross, y_std_cross, fmt='b-', label='Validation')
    plt.title('Prediction Plot')

    plt.subplot(1, 2, 2)
    plt.errorbar(t_train, np.zeros(len(mag_train)), magerr_train, fmt='k.', markersize=10)
    plt.errorbar(t_cross, np.zeros(len(mag_train)), magerr_cross, fmt='k.', markersize=10)
    plt.errorbar(t_pred_train, y_pred_train[:, 0] - mag_train[WINDOW_LEN+1: -1], y_std_train[:, 0], fmt='g-', label='Training')
    plt.errorbar(t_pred_cross, y_pred_cross[:, 0] - mag_cross[WINDOW_LEN+1: -1], y_std_cross[:, 0], fmt='b-', label='Validation')
    plt.title('Residual Plot')

    return dis_fig


def continuous_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_std_train, y_pred_train_n,
                    t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_std_cross, y_pred_cross_n):

    con_fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))
    for ax in axes.ravel():
        ax.xlim(t_train.min(), t_cross.max())
        ax.errorbar(t_train, mag_train, magerr_train, fmt='k.')
        ax.errorbar(t_train, mag_cross, magerr_cross, fmt='k.')
        ax.set_xlabel('MJD')
        ax.set_ylabel('Mag')
        plt.legend()

    plt.subplot(1, 2, 1)
    n_paths = np.shape(y_pred_train_n)[0]
    for i in range(n_paths):
        plt.plot(t_pred_train, y_pred_train_n[i], color='green', alpha=(1 - i / float(n_paths)))
        plt.plot(t_pred_cross, y_pred_cross_n[i], color='blue', alpha=(1 - i / float(n_paths)))
    plt.title('Sample Plot')

    plt.subplot(1, 2, 2)
    plt.plot(t_pred_train, y_pred_train, color='green', ls='-')
    plt.plot(t_pred_cross, y_pred_cross, color='blue', ls='-')
    plt.fill_between(t_pred_train, y1=y_pred_train + np.sqrt(y_std_train), y2=y_pred_train - np.sqrt(y_std_train),
                     color='LimeGreen', alpha=0.5)
    plt.fill_between(t_pred_cross, y1=y_pred_cross + np.sqrt(y_std_cross), y2=y_pred_cross - np.sqrt(y_std_cross),
                     color='DodgerBlue', alpha=0.5)
    plt.title('Average Plot')

    return con_fig


def match_list(t, t_pred, y_pred, y_std):
    t_pred_match = []
    y_pred_match = []
    y_pred_var_match = []
    for time in t:
        t_id = min(range(len(t_pred)), key=lambda i: abs(t_pred[i] - time))
        t_pred_match.append(t_pred[t_id])
        y_pred_match.append(y_pred[t_id])
        y_pred_var_match.append(y_std[t_id])

    return t_pred_match, y_pred_match, y_pred_var_match


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
