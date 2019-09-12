import numpy as np

def delta_list(raw_list):
    delta_list = []
    for i in range(1, len(raw_list)):
        delta = raw_list[i] - raw_list[i - 1]
        delta_list.append(delta)
    delta_list = np.array(delta_list)

    return delta_list