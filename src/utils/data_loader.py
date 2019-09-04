import pickle
import numpy as np
import pandas as pd
import os
from global_setting import DATA_FOLDER


def data_loader(crts_id):
    file_path = os.path.join(DATA_FOLDER, str(crts_id) + '.csv')
    with open(file_path) as handle:
        content = pd.read_csv(handle)
        mag_list = np.array(content['Mag'])
        magerr_list = np.array(content['Magerr'])
        mjd_list = np.array(content['MJD'])

        return mag_list, magerr_list, mjd_list
