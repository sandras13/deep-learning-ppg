import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import preprocessing
import train_with_mlflow as ml

params = {
    "seed" : 256,
    "learning_rate" : 0.001,
    "weight_decay" : 1e-6,
    "step_size" : 3,
    "gamma" : 0.85,
    "batch_size" : 128,
    "epochs" : 30,
    "num_resblocks" : 5,
    "num_lstm_layers" : 2,
}

np.random.seed(params["seed"])
torch.manual_seed(params["seed"])

features, targets, subj_data, labels, fs, avg = preprocessing.load_data(dataset_id = 2)
params["window_size"] = fs * 8
params["overlap"] = params["window_size"] * 7//8
scaled_data = preprocessing.scale_data(features, targets)

sliding_X_data, sliding_y_data = preprocessing.apply_sliding_window(scaled_data, targets, subj_data,
                                                         params["window_size"], params["overlap"], avg)

X_data = sliding_X_data.astype(np.float32)
y_data = sliding_y_data.astype(np.uint8)

print(X_data.shape, y_data.shape)

params["num_channels"] = X_data.shape[2]
params["num_classes"] = len(labels)

architecure_id = 1 # LSTM - 1, ResNet - 2

ml.mlflow_training_loop(X_data, y_data, params, architecure_id, crossvalid=False)