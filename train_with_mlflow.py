import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import architectures as arch
import mlflow
import mlflow.pytorch
import preprocessing
from sklearn.model_selection import KFold
import network_training

def log_params(params):
    mlflow.log_param("Batch size", params["batch_size"])
    mlflow.log_param("Learning rate", params["learning_rate"])
    mlflow.log_param("Step size", params["step_size"])
    mlflow.log_param("Gamma", params["gamma"])
    mlflow.log_param("Weight decay", params["weight_decay"])
    mlflow.log_param("Epochs", params["epochs"])
    mlflow.log_param("Number of ResBlocks", params["num_resblocks"])
    mlflow.log_param("Number of LSTM layers", params["num_lstm_layers"])

def log_metrics(avg_loss_train, accuracy_train, recall_train, f1_train, precision_train,
                avg_loss_val, accuracy_val, recall_val, f1_val, precision_val, epoch):
    
    mlflow.log_metric("Loss - train", avg_loss_train, step = epoch)
    mlflow.log_metric("Accuracy - train", accuracy_train, step = epoch)
    mlflow.log_metric("Recall - train", recall_train, step = epoch)
    mlflow.log_metric("F1 score - train", f1_train, step = epoch)
    mlflow.log_metric("Precision - train", precision_train, step = epoch)

    mlflow.log_metric("Loss - valid", avg_loss_val, step = epoch)
    mlflow.log_metric("Accuracy - valid", accuracy_val, step = epoch)
    mlflow.log_metric("Recall - valid", recall_val, step = epoch)
    mlflow.log_metric("F1 score - valid", f1_val, step = epoch)
    mlflow.log_metric("Precision - valid", precision_val, step = epoch)

def log_results(mean_avg_losses_train, mean_f1_scores_train, mean_accuracies_train, 
                mean_recalls_train, mean_precisions_train, mean_avg_losses_val, 
                mean_f1_scores_val, mean_accuracies_val, mean_recalls_val, 
                mean_precisions_val):
    
    mlflow.log_param("Average loss - train", mean_avg_losses_train)
    mlflow.log_param("F1 score - train", mean_f1_scores_train)
    mlflow.log_param("Accuracy - train", mean_accuracies_train)
    mlflow.log_param("Recall - train", mean_recalls_train)
    mlflow.log_param("Precision - train", mean_precisions_train)

    mlflow.log_param("Average loss - val", mean_avg_losses_val)
    mlflow.log_param("F1 score - val", mean_f1_scores_val)
    mlflow.log_param("Accuracy - val", mean_accuracies_val)
    mlflow.log_param("Recall - val", mean_recalls_val)
    mlflow.log_param("Precision - val", mean_precisions_val)

def mlflow_training_loop(X_data, y_data, params, architecure_id, crossvalid = False):
    mlflow.set_tracking_uri("http://localhost:5000")

    if crossvalid:
        num_folds = 5
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=100)
        data_splits = [(X_data[train_ind], X_data[val_ind], y_data[train_ind], y_data[val_ind]) for train_ind, val_ind in kf.split(X_data)]
    else:
        X_train, X_val, y_train, y_val = network_training.split_data(X_data, y_data, train_size=0.8)
        data_splits = [(X_train, X_val, y_train, y_val)]

    avg_losses_train = []
    f1_scores_train = []
    accuracies_train = []
    recalls_train = []
    precisions_train = []

    avg_losses_val = []
    f1_scores_val = []
    accuracies_val = []
    recalls_val = []
    precisions_val = []

    with mlflow.start_run():
        log_params(params)

        for fold, (X_train_fold, X_val_fold, y_train_fold, y_val_fold) in enumerate(data_splits):
            print(f"Fold {fold+1}/{num_folds if crossvalid else 1}")
            if architecure_id == 1:
                network = arch.DeepConvLSTM(params["num_channels"], params["num_classes"], num_layers=params["num_lstm_layers"], hidden_size=256)
            else:
                network = arch.ResNet(params["num_channels"], params["num_classes"], num_resblocks=params["num_resblocks"])
            
            if fold == 0: print(network)

            optimizer = torch.optim.Adam(network.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
            class_weights = preprocessing.get_class_weights(y_data)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])

            trainloader, valloader = network_training.data_loader(X_train_fold, y_train_fold,
                                                                  X_val_fold, y_val_fold, params["batch_size"])
            
            for epoch in range(params["epochs"]):
                print(f"Epoch {epoch + 1}:")
                avg_loss_train, f1_train, accuracy_train, recall_train, precision_train = \
                    network_training.train(trainloader, network, optimizer, criterion, scheduler)
                avg_loss_val, f1_val, accuracy_val, recall_val, precision_val = \
                    network_training.valid(valloader, network, criterion)
            
                if not crossvalid:
                    log_metrics(avg_loss_train, accuracy_train, recall_train, f1_train, precision_train,
                        avg_loss_val, accuracy_val, recall_val, f1_val, precision_val, epoch)
                
            avg_losses_train.append(avg_loss_train)
            f1_scores_train.append(f1_train)
            accuracies_train.append(accuracy_train)
            recalls_train.append(recall_train)
            precisions_train.append(precision_train)

            avg_losses_val.append(avg_loss_val)
            f1_scores_val.append(f1_val)
            accuracies_val.append(accuracy_val)
            recalls_val.append(recall_val)
            precisions_val.append(precision_val)

        mean_avg_losses_train = np.mean(avg_losses_train)
        mean_f1_scores_train = np.mean(f1_scores_train)
        mean_accuracies_train = np.mean(accuracies_train)
        mean_recalls_train = np.mean(recalls_train)
        mean_precisions_train = np.mean(precisions_train)

        mean_avg_losses_val = np.mean(avg_losses_val)
        mean_f1_scores_val = np.mean(f1_scores_val)
        mean_accuracies_val = np.mean(accuracies_val)
        mean_recalls_val = np.mean(recalls_val)
        mean_precisions_val = np.mean(precisions_val)

        if crossvalid: log_results(mean_avg_losses_train, mean_f1_scores_train,
                                   mean_accuracies_train, mean_recalls_train,
                                   mean_precisions_train, mean_avg_losses_val, 
                                   mean_f1_scores_val, mean_accuracies_val,
                                   mean_recalls_val, mean_precisions_val)

            