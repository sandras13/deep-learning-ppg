import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import module as hm
import torch.optim.lr_scheduler as lr_scheduler
import architectures as arch
import mlflow
import mlflow.pytorch
from sklearn.model_selection import KFold
import net_training as nt

np.random.seed(256)
torch.manual_seed(256)

features, targets, subj_data, labels, fs, window_size, overlap, avg = hm.import_data(dataset_id = 1)
scaled_data = hm.scale_data(features, targets)

sliding_X_data, sliding_y_data = hm.apply_sliding_window(scaled_data, targets, subj_data, window_size, overlap, avg)

X_data = sliding_X_data.astype(np.float32)
y_data = sliding_y_data.astype(np.uint8)
hm.plot_data(y_data, labels)

print(X_data.shape, y_data.shape)

num_channels = X_data.shape[2]
num_classes = len(labels)

#nt.crossvalid(X_data, y_data, num_channels, num_classes)
nt.classic_train(X_data, y_data, num_channels, num_classes)