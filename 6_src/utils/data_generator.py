import numpy as np
import os
import pandas as pd
data_folder = '/Users/mingyu/Desktop/dataset_2/raw_data'

def generate_sin(lim, amp, n, name):
    t = np.sort(np.random.uniform(0, lim, n))
    mag = amp * np.sin((t/lim)*20)
    mag_err = np.random.normal(0, amp/20, n)
    content = list(zip(t, mag, mag_err))
    columns = ['MJD', 'Mag', 'Magerr']
    full_frame = pd.DataFrame(content, columns=columns)
    full_frame.to_csv(os.path.join(data_folder, name + '.csv'))

def generate_sqrt(lim, amp, n, name):
    t = np.sort(np.random.uniform(0, lim, n))
    mag = amp * np.square((t/lim)*20)
    mag_err = np.random.normal(0, amp/20, n)
    content = list(zip(t, mag, mag_err))
    columns = ['MJD', 'Mag', 'Magerr']
    full_frame = pd.DataFrame(content, columns=columns)
    full_frame.to_csv(os.path.join(data_folder, name + '.csv'))

if __name__ == '__main__':
    for i in range(5):
        generate_sin(lim=3000, amp=20, n=300, name=str(i+1))
        # generate_sqrt(lim=3000, amp=1, n=300, name=str(i + 1))
        pass
