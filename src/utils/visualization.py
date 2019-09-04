import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from global_setting import DATA_FOLDER
from utils.data_loader import data_loader

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
