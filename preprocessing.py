import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

def resample_signal(data, original_sampling_rate, target_sampling_rate):
    original_time_axis = np.arange(0, len(data)) / original_sampling_rate
    target_time_axis = np.arange(0, original_time_axis[-1], 1 / target_sampling_rate)
    resampled_data = signal.resample(data, int(len(target_time_axis)))

    return resampled_data

def load_WESAD(dataset_dir):
    subj_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17] 
    data = np.empty((0, 3))

    for subject in subj_list:
        obj = pd.read_pickle(f'{dataset_dir}\S{subject}\S{subject}.pkl')
        ppg = np.array(obj['signal']['wrist']['BVP'])
        y_lab = obj['label']
        y_lab = resample_signal(y_lab, 700, 64)
        y_lab = np.rint(y_lab)
        y_lab = y_lab.astype(np.uint8).reshape(-1, 1)
        subj_ind = np.full((len(ppg), 1), subject)

        fusion = np.hstack((subj_ind, ppg, y_lab))
        data = np.vstack((data, fusion))

    condition = ((data[:, -1] >= 1) & (data[:, -1] <= 4))
    data = data[condition]

    subj_data = data[:, 0]
    x_data = data[:, 1].reshape(-1, 1)
    y_data = data[:, -1] - 1

    return x_data, y_data, subj_data

def load_AffectiveROAD(dataset_dir, wrist):
    # wrist: 0 - both, 1 - left, 2 - right

    if wrist != 0: x_data = np.empty((0, 1))
    else: x_data = np.empty((0, 2))

    y_data = np.empty((0, 1))
    subj_data = np.empty((0, 1))
    subj_metric_timestamps = pd.read_csv(f'{dataset_dir}\Subj_metric\Annot_Subjective_metric.csv')

    route_start = 'Z_Start'
    route_end = 'Z_End.1'

    start_vals_SM = np.array(subj_metric_timestamps[route_start])
    end_vals_SM = np.array(subj_metric_timestamps[route_end])

    left_wrist_timestamps = pd.read_csv(f'{dataset_dir}\E4\Annot_E4_Left.csv')
    right_wrist_timestamps = pd.read_csv(f'{dataset_dir}\E4\Annot_E4_Right.csv')

    end_vals_LW = np.array(left_wrist_timestamps[route_end])
    end_vals_RW = np.array(right_wrist_timestamps[route_end])

    subj_list = [i for i in range(13)]

    for ind in subj_list:
        if ind == 1:
            lw_ppg1 = pd.read_csv(f'{dataset_dir}\E4\{ind + 1}-E4-Drv{ind + 1}\Left1\BVP.csv')
            lw_ppg2 = pd.read_csv(f'{dataset_dir}\E4\{ind + 1}-E4-Drv{ind + 1}\Left2\BVP.csv')
            lw_ppg = np.vstack((lw_ppg1, lw_ppg2))
        else:
            lw_ppg = pd.read_csv(f'{dataset_dir}\E4\{ind + 1}-E4-Drv{ind + 1}\Left\BVP.csv')
            lw_ppg = np.array(lw_ppg)

        rw_ppg = pd.read_csv(f'{dataset_dir}\E4\{ind + 1}-E4-Drv{ind + 1}\Right\BVP.csv')
        rw_ppg = np.array(rw_ppg)

        sm = pd.read_csv(f'{dataset_dir}\Subj_metric\SM_Drv{ind + 1}.csv')
        sm = np.array(sm)

        if(end_vals_SM[ind] > len(sm)):
            end_vals_SM[ind] = len(sm)

        seq_length = end_vals_SM[ind] - start_vals_SM[ind] - 1

        lw_ppg = lw_ppg[(end_vals_LW[ind] - seq_length)*16:end_vals_LW[ind]*16]
        rw_ppg = rw_ppg[(end_vals_RW[ind] - seq_length)*16:end_vals_RW[ind]*16]
        sm = sm[start_vals_SM[ind]:end_vals_SM[ind]]
        
        sm = resample_signal(sm, 4, 64)

        subj_ind = np.full((len(lw_ppg), 1), ind)
        fusion = np.hstack((lw_ppg, rw_ppg))

        if wrist == 0: x_data = np.vstack((x_data, fusion))
        elif wrist == 1: x_data = np.vstack((x_data, lw_ppg))
        else: x_data = np.vstack((x_data, rw_ppg))

        y_data = np.vstack((y_data, sm))
        subj_data = np.vstack((subj_data, subj_ind))
    
    y_data = y_data.reshape(-1, )
    return x_data, y_data, subj_data

def load_data(dataset_id, wrist = None):
    if dataset_id == 1:
        labels = ('baseline', 'stress', 'amusement', 'meditation')
        dataset_dir = 'WESAD'
        x_data, y_data, subj_data = load_WESAD(dataset_dir)
        fs = 64
        avg = False

    elif dataset_id == 2:
        labels = ('low', 'medium', 'high')
        dataset_dir = 'AffectiveROAD_Data\Database'
        x_data, y_data, subj_data = load_AffectiveROAD(dataset_dir, wrist)
        fs = 64
        avg = True

    return x_data, y_data, subj_data, labels, fs, avg

def scale_data(x_data, y_data):
    scaled_data = StandardScaler().fit_transform(x_data, y_data)
    return scaled_data

def get_class_weights(y_data):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_data), y=y_data)
    class_weights = torch.FloatTensor(class_weights)

    return class_weights

def map_values(val):
    if val < 0.4: return 0
    if val < 0.75: return 1
    return 2

def apply_sliding_window(features, targets, subj_data, window_size, overlap, avg = False):
    sliding_X_data = []
    sliding_y_data = []
    i = 0

    while i < len(features) - window_size:
        window_X = features[i:i + window_size]
        window_y = targets[i:i + window_size]
        subj_window = subj_data[i:i + window_size]

        if len(np.unique(subj_window) == 1):
            if avg:
                y_avg = np.mean(window_y)
                y_mapped = map_values(y_avg)
                sliding_X_data.append(window_X)
                sliding_y_data.append(y_mapped)

            elif len(np.unique(window_y) == 1):
                sliding_X_data.append(window_X)
                sliding_y_data.append(window_y[-1])

        i += (window_size - overlap)
    
    sliding_X_data = np.array(sliding_X_data)
    sliding_y_data = np.array(sliding_y_data).reshape(-1, )
    return sliding_X_data, sliding_y_data