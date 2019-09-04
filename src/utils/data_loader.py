import pickle
import math
import random
import numpy as np
import pandas as pd
import os
from global_setting import DATA_FOLDER
import json
np.random.seed(1)


class DataLoader():

    def __init__(self, crts_id, is_short=False):
        self.crts_id = crts_id
        self.is_short = is_short
        self.data_config = json.load(open('./config/data_config.json'))
        self.model_config = json.load(open('./config/model_config.json'))

    def load_raw_data(self):
        file_path = os.path.join(DATA_FOLDER, str(self.crts_id) + '.csv')
        with open(file_path) as handle:
            content = pd.read_csv(handle)
            mag_list_ = np.array(content['Mag'])
            magerr_list_ = np.array(content['Magerr'])
            mjd_list_ = np.array(content['MJD'])

            # Sort data
            index_list = np.argsort(mjd_list_)
            mag_list = mag_list_[index_list]
            magerr_list = magerr_list_[index_list]
            mjd_list = mjd_list_[index_list]

            if not self.is_short:
                return mag_list, magerr_list, mjd_list

            else:
                short_len = self.data_config["short_len"]
                return mag_list[:short_len], magerr_list[:short_len], mjd_list[:short_len]


    '''
    def random_index_partition(self, length):
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
    '''

    def partition_sequential_index(self):
        mag_list, magerr_list, mjd_list = self.load_raw_data()
        train_percent = self.data_config["partition"]["train"]
        cross_percent = self.data_config["partition"]["cross"]
        test_percent = self.data_config["partition"]["test"]
        length = len(mag_list)
        p1 = int(math.floor(length * train_percent))
        p2 = int(math.floor(length * (train_percent + cross_percent)))

        index_list = list(range(length))
        index_train = index_list[0:p1]
        index_cross = index_list[p1:p2]
        index_test = index_list[p2:]

        return index_train, index_cross, index_test

    def load_partition_data(self):
        mag_list, magerr_list, mjd_list = self.load_raw_data()
        index_train, index_cross, index_test = self.partition_sequential_index()
        t_list = mjd_list - mjd_list.min()

        mag_list_train = mag_list[index_train]
        mag_list_cross = mag_list[index_cross]
        mag_list_test = mag_list[index_test]
        magerr_list_train = magerr_list[index_train]
        magerr_list_cross = magerr_list[index_cross]
        magerr_list_test = magerr_list[index_test]
        t_list_train = t_list[index_train]
        t_list_cross = t_list[index_cross]
        t_list_test = t_list[index_test]

        data_dict = {'mag_list_train': mag_list_train, 'mag_list_cross': mag_list_cross, 'mag_list_test': mag_list_test,
                     'magerr_list_train': magerr_list_train, 'magerr_list_cross': magerr_list_cross,
                     'magerr_list_test': magerr_list_test, 't_list_train': t_list_train, 't_list_cross': t_list_cross,
                     't_list_test': t_list_test, 'crts_id': self.crts_id}

        return data_dict
