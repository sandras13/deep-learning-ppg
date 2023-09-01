import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from scipy import signal
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
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

    x_data = data[:, 1].reshape(-1, 1)
    y_data = data[:, -1] - 1

    return x_data, y_data

def load_AffectiveROAD():
    x_data = np.empty((0, 2))
    y_data = np.empty((0, 1))
    subj_metric_timestamps = pd.read_csv('AffectiveROAD_Data\Database\Subj_metric\Annot_Subjective_metric.csv')

    start_vals_SM = np.array(subj_metric_timestamps['Z_Start'])
    end_vals_SM = np.array(subj_metric_timestamps['Z_End.1'])

    seq_length = end_vals_SM - start_vals_SM

    left_wrist_timestamps = pd.read_csv('AffectiveROAD_Data\Database\E4\Annot_E4_Left.csv')
    right_wrist_timestamps = pd.read_csv('AffectiveROAD_Data\Database\E4\Annot_E4_Right.csv')

    end_vals_LW = np.array(left_wrist_timestamps['Z_End.1'])
    end_vals_RW = np.array(right_wrist_timestamps['Z_End.1'])

    start_vals_LW = end_vals_LW - seq_length
    start_vals_RW = end_vals_RW - seq_length

    subj_list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

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
        sm = np.repeat(sm, 16, axis = 0)

        #subj_ind = np.full((len(lw_ppg), 1), ind)
        fusion = np.hstack((lw_ppg, rw_ppg))
        x_data = np.vstack((x_data, fusion))
        y_data = np.vstack((y_data, sm))
        
    return x_data, y_data

def import_data(dataset_id):
    if dataset_id == 1:
        labels = ('baseline', 'stress', 'amusement', 'meditation')
        dataset_dir = 'WESAD'
        x_data, y_data = load_WESAD(dataset_dir)
        fs = 64
        window_size = 512
        overlap = 256
        avg = False

    elif dataset_id == 2:
        labels = ('low', 'medium', 'high')
        dataset_dir = 'AffectiveROAD'
        x_data, y_data = load_AffectiveROAD()
        fs = 64
        window_size = 512
        overlap = 256
        avg = True

    return x_data, y_data, labels, fs, window_size, overlap, avg

def scale_data(x_data, y_data):
    scaled_data = StandardScaler().fit_transform(x_data, y_data)
    return scaled_data

def filter_data(data, fs):
    fnyq = fs / 2

    filt = signal.butter(N = 2, Wn = 0.5 / fnyq, btype='high', fs = fs, output='sos', analog = False)
    filt_data = signal.sosfilt(filt, data)

    return filt_data

def plot_data(y_data, labels):
    class_counts = np.bincount(y_data)
    plt.bar(labels, class_counts)
    plt.show()

def get_class_weights(y_data):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_data), y=y_data)
    class_weights = torch.FloatTensor(class_weights)

    return class_weights

def map_values(val):
    if val < 0.35: return 0
    if val < 0.7: return 1
    return 2

def apply_sliding_window(features, targets, window_size, overlap, avg = False):
    sliding_X_data = []
    sliding_y_data = []
    i = 0

    while i < len(features) - window_size:
        window_X = features[i:i + window_size]
        window_y = targets[i:i + window_size]

        if avg:
            y_avg = np.mean(window_y)
            y_mapped = map_values(y_avg)
            if y_mapped != 10:
                sliding_X_data.append(window_X)
                sliding_y_data.append(y_mapped)
            else:
                print(window_y)

        elif len(np.unique(window_y) == 1):
            sliding_X_data.append(window_X)
            sliding_y_data.append(window_y[-1])

        i += (window_size - overlap)
    
    sliding_X_data = np.array(sliding_X_data)
    sliding_y_data = np.array(sliding_y_data).reshape(-1, )
    return sliding_X_data, sliding_y_data

def split_data(X_data, y_data, train_size):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, train_size=train_size)
    return X_train, X_val, y_train, y_val

def data_loader(X_train, y_train, X_valid, y_valid, batch_size):
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader

def print_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"Layer: {name}\t Shape: {param.shape}\t Gradient: {param.grad.mean()}")

def print_score(preds, true):
    f1 = f1_score(true, preds, average='macro')
    acc = accuracy_score(true, preds)
    rec = recall_score(true, preds, average='macro')
    prec = precision_score(true, preds, average='macro')
    print(f1, acc, rec, prec)
    return f1, acc, rec, prec

def train(data_loader, model, optimizer, criterion, scheduler = None):
    model.train()
    train_preds = []
    train_gt = []

    print("Training: ")

    total_loss = 0
    batch_num = 1
    cnt = 10

    for inputs, targets in data_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        y_preds = np.argmax(outputs.detach().numpy(), axis=-1)
        y_true = targets.numpy().flatten()
            
        total_loss += loss.item()
        #if batch_num % cnt == 0:
        #    print(f"Average loss: {total_loss / cnt:.4f}")
        #    total_loss = 0
        #    print_gradients(model)

        train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
        train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))

        batch_num += 1

    if scheduler is not None:
        print(f"Current learning rate: {scheduler.get_last_lr()}")
        scheduler.step()
        
    f1_train, accuracy_train, recall_train, precision_train = print_score(train_preds, train_gt)
    #print(train_preds, train_gt)

    avg_loss = total_loss / batch_num

    return avg_loss, f1_train, accuracy_train, recall_train, precision_train
        
def valid(data_loader, model, criterion):
    model.eval()

    val_preds = []
    val_gt = []

    print("Validation: ")

    total_loss = 0
    batch_num = 1

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            y_preds = np.argmax(outputs.detach().numpy(), axis=-1)
            y_true = targets.numpy().flatten()
            val_preds = np.concatenate((np.array(val_preds, int), np.array(y_preds, int)))
            val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))

            total_loss += loss.item()
            batch_num += 1

    #print(val_preds, val_gt)
    f1_val, accuracy_val, recall_val, precision_val = print_score(val_preds, val_gt)
    avg_loss = total_loss / batch_num
    
    return avg_loss, f1_val, accuracy_val, recall_val, precision_val
