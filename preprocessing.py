import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.signal
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import resample

def resample_signal(data, original_sampling_rate, target_sampling_rate):
    original_time_axis = np.arange(0, len(data)) / original_sampling_rate
    target_time_axis = np.arange(0, original_time_axis[-1], 1 / target_sampling_rate)
    resampled_data = resample(data, int(len(target_time_axis)))

    return resampled_data

def load_WESAD(dataset_dir):
    subj_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17] 
    data = np.empty((0, 3))

    for subject in subj_list:
        obj = pd.read_pickle(f'{dataset_dir}/S{subject}/S{subject}.pkl')
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

def load_AffectiveROAD():
    x_data = np.empty((0, 2))
    y_data = np.empty((0, 1))
    subj_data = np.empty((0, 1))
    subj_metric_timestamps = pd.read_csv('AffectiveROAD_Data\Database\Subj_metric\Annot_Subjective_metric.csv')

    start_vals_SM = np.array(subj_metric_timestamps['Z_End'])
    end_vals_SM = np.array(subj_metric_timestamps['Z_Start.1'])

    seq_length = end_vals_SM - start_vals_SM

    left_wrist_timestamps = pd.read_csv('AffectiveROAD_Data\Database\E4\Annot_E4_Left.csv')
    right_wrist_timestamps = pd.read_csv('AffectiveROAD_Data\Database\E4\Annot_E4_Right.csv')

    end_vals_LW = np.array(left_wrist_timestamps['Z_Start.1'])
    end_vals_RW = np.array(right_wrist_timestamps['Z_Start.1'])

    start_vals_LW = end_vals_LW - seq_length
    start_vals_RW = end_vals_RW - seq_length

    subj_list = [i for i in range(13)]

    start_vals_LW *= 16
    end_vals_LW *= 16
    start_vals_RW *= 16
    end_vals_RW *= 16

    for ind in subj_list:
        lw_ppg = pd.read_csv(f'AffectiveROAD_Data\Database\E4\{ind + 1}-E4-Drv{ind + 1}\Left\BVP.csv')
        lw_ppg = lw_ppg[start_vals_LW[ind]:end_vals_LW[ind]]

        rw_ppg = pd.read_csv(f'AffectiveROAD_Data\Database\E4\{ind + 1}-E4-Drv{ind + 1}\Right\BVP.csv')
        rw_ppg = rw_ppg[start_vals_RW[ind]:end_vals_RW[ind]]

        sm = pd.read_csv(f'AffectiveROAD_Data\Database\Subj_metric\SM_Drv{ind + 1}.csv')
        sm = sm[start_vals_SM[ind]:end_vals_SM[ind]]
        sm = np.array(sm)
        sm = resample_signal(sm, 4, 64)

        subj_ind = np.full((len(lw_ppg), 1), ind)
        fusion = np.hstack((lw_ppg, rw_ppg))

        # x_data = np.vstack((x_data, rw_ppg))
        x_data = np.vstack((x_data, fusion))
        y_data = np.vstack((y_data, sm))
        subj_data = np.vstack((subj_data, subj_ind))
        
    return x_data, y_data, subj_data

def load_data(dataset_id):
    if dataset_id == 1:
        labels = ('baseline', 'stress', 'amusement', 'meditation')
        dataset_dir = 'WESAD'
        x_data, y_data, subj_data = load_WESAD(dataset_dir)
        fs = 64
        avg = False

    elif dataset_id == 2:
        labels = ('low', 'medium', 'high')
        x_data, y_data, subj_data = load_AffectiveROAD()
        fs = 64
        avg = True

    return x_data, y_data, subj_data, labels, fs, avg

def scale_data(x_data, y_data):
    scaled_data = StandardScaler().fit_transform(x_data, y_data)
    return scaled_data

def filter_data(data, fs):
    fnyq = fs / 2

    filt = signal.butter(N = 2, Wn = 0.5 / fnyq, btype='high', fs = fs, output='sos', analog = False)
    filt_data = signal.sosfilt(filt, data)

    return filt_data

def get_class_weights(y_data):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_data), y=y_data)
    class_weights = torch.FloatTensor(class_weights)

    return class_weights

def map_values(val, num_classes):
    if num_classes == 3:
        if val < 0.45: return 0
        if val < 0.75: return 1
        return 2
    
    if num_classes == 2:
        if val < 0.75: return 0
        return 1

    # if val <= 0.33: return 0
    # if val >= 0.67: return 1
    # return 2

def apply_sliding_window(features, targets, subj_data, window_size, overlap, avg = False, num_classes = None):
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
                y_mapped = map_values(y_avg, num_classes)
                sliding_X_data.append(window_X)
                sliding_y_data.append(y_mapped)

            elif len(np.unique(window_y) == 1):
                sliding_X_data.append(window_X)
                sliding_y_data.append(window_y[-1])

        i += (window_size - overlap)
    
    sliding_X_data = np.array(sliding_X_data)
    sliding_y_data = np.array(sliding_y_data).reshape(-1, )
    return sliding_X_data, sliding_y_data