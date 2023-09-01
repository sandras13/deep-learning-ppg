import numpy as np
import torch
import torch.nn as nn
import module as hm
import torch.optim.lr_scheduler as lr_scheduler
import architectures as arch
import mlflow
import mlflow.pytorch
from sklearn.model_selection import KFold

def crossvalid(X_data, y_data, num_channels, num_classes):
    mlflow.set_tracking_uri("http://localhost:5000")
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=100)

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
        learning_rate = 0.001
        weight_decay = 5e-4
        step_size = 3
        gamma = 0.8
        batch_size = 128
        epochs = 30

        mlflow.log_param("Batch size", batch_size)
        mlflow.log_param("Learning rate", learning_rate)
        mlflow.log_param("Step size", step_size)
        mlflow.log_param("Gamma", gamma)
        mlflow.log_param("Weight decay", weight_decay)
        mlflow.log_param("Epochs", epochs)

        for fold, (train_ind, val_ind) in enumerate(kf.split(X_data)):
            print(f"Fold {fold+1}/{num_folds}")

            #network = arch.DeepConvLSTM(num_channels, num_classes,  num_layers = 2, hidden_size = 128)
            network = arch.ResNet(num_channels = num_channels, num_classes = num_classes)
            #network = arch.MultiPathNet(num_channels, num_classes)
            #network = arch.SimpleCNN(num_channels, num_classes, 64)

            if fold == 0: print(network)

            X_train_fold = X_data[train_ind]
            y_train_fold = y_data[train_ind]
            X_val_fold = X_data[val_ind]
            y_val_fold = y_data[val_ind]

            scheduler = None

            optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
            class_weights = hm.get_class_weights(y_data)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            trainloader, valloader = hm.data_loader(X_train_fold, y_train_fold, X_val_fold, y_val_fold, batch_size)
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}:")
                avg_loss_train, f1_train, accuracy_train, recall_train, precision_train = \
                    hm.train(trainloader, network, optimizer, criterion, scheduler)
                avg_loss_val, f1_val, accuracy_val, recall_val, precision_val = \
                    hm.valid(valloader, network, criterion)
            
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