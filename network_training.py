import numpy as np
import torch
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

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

def split_data(X_data, y_data, train_size):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, train_size=train_size)
    return X_train, X_val, y_train, y_val

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
        # if batch_num % cnt == 0:
        #     print(f"Average loss: {total_loss / cnt:.4f}")
        #     total_loss = 0
        #     print_gradients(model)

        train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
        train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))

        batch_num += 1

    if scheduler is not None:
        print(f"Current learning rate: {scheduler.get_last_lr()}")
        scheduler.step()
        
    f1_train, accuracy_train, recall_train, precision_train = print_score(train_preds, train_gt)
    # print(train_preds, train_gt)

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

    # print(val_preds, val_gt)
    f1_val, accuracy_val, recall_val, precision_val = print_score(val_preds, val_gt)
    avg_loss = total_loss / batch_num
    
    return avg_loss, f1_val, accuracy_val, recall_val, precision_val



