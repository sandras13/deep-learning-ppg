import torch
import torch.nn as nn
import numpy as np

class DeepConvLSTM(nn.Module):
    def __init__(self, num_channels, num_classes, num_layers, hidden_size):
        super(DeepConvLSTM, self).__init__()
        self.avgpool = nn.AvgPool1d(kernel_size=4, stride=4)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.avgpool(x)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1) 
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        
        return out

class MultiPathResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MultiPathResidualBlock, self).__init__()

        self.conv1_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1_1 = nn.BatchNorm1d(out_channels)

        self.conv1_3 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn1_3 = nn.BatchNorm1d(out_channels)

        self.conv1_5 = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4)
        self.bn1_5 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels))
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        out_3 = self.relu(self.bn1_3(self.conv1_3(x)))
        out_5 = self.relu(self.bn1_5(self.conv1_5(x)))

        out = out_1 + out_3 + out_5

        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bottleneck=False):
        super(ResidualBlock, self).__init__()
        self.bottleneck = bottleneck
        if bottleneck:
            self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm1d(out_channels // 4)
            self.conv2 = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels // 4)
            self.conv3 = nn.Conv1d(out_channels // 4, out_channels, kernel_size=1, stride=1)
            self.bn3 = nn.BatchNorm1d(out_channels)

        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if bottleneck:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels)
                )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.bottleneck:
            out = self.relu(out)
            out = self.bn3(self.conv3(out))
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ResNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_filters = 64

        self.avgpool1 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.conv1 = nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(self.num_filters)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.resblock1 = ResidualBlock(self.num_filters, self.num_filters, bottleneck=False)
        self.resblock2 = ResidualBlock(self.num_filters, self.num_filters, bottleneck=False)
        self.resblock3 = ResidualBlock(self.num_filters, self.num_filters, bottleneck=False)
        self.resblock4 = ResidualBlock(self.num_filters, self.num_filters, bottleneck=False)
        self.resblock5 = ResidualBlock(self.num_filters, self.num_filters, bottleneck=False)
        self.resblock6 = ResidualBlock(self.num_filters, self.num_filters, bottleneck=False)
        self.resblock7 = ResidualBlock(self.num_filters, self.num_filters, bottleneck=False)

        #self.resblock1 = MultiPathResidualBlock(self.num_filters, self.num_filters)
        #self.resblock2 = MultiPathResidualBlock(self.num_filters, self.num_filters)
        #self.resblock3 = MultiPathResidualBlock(self.num_filters, self.num_filters)
        #self.resblock4 = MultiPathResidualBlock(self.num_filters, self.num_filters)
        #self.resblock5 = MultiPathResidualBlock(self.num_filters, self.num_filters)

        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.num_filters, self.num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.avgpool1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)

        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class InceptionModule1D(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5):
        super(InceptionModule1D, self).__init__()
        
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm1d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_3x3),
            nn.ReLU(inplace=True)
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class MultiPathNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(MultiPathNet, self).__init__()

        self.avgpool1 = nn.AvgPool1d(4, 4)
        
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule1D(32, 32, 32, 32)
        self.inception3b = InceptionModule1D(128, 128, 128, 128)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(512, 32, 3)
        self.bn3 = nn.BatchNorm1d(32)

        self.inception4a = InceptionModule1D(32, 32, 32, 32)
        self.inception4b = InceptionModule1D(128, 128, 128, 128)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(512, 32, 3)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.inception5a = InceptionModule1D(32, 32, 32, 32)
        self.inception5b = InceptionModule1D(128, 128, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.avgpool1(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.avgpool2(x)
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.inception4a(x)
        x = self.inception4b(x)

        x = self.avgpool3(x)
        x = self.relu(self.bn4(self.conv4(x)))

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes, num_filters):
        super(SimpleCNN, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.avgpool1 = nn.AvgPool1d(4, 4)

        self.conv1 = nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.num_filters)

        self.conv2 = nn.Conv1d(in_channels=self.num_filters, out_channels=self.num_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(self.num_filters * 2)

        self.conv3 = nn.Conv1d(in_channels=self.num_filters * 2, out_channels=self.num_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(self.num_filters * 4)

        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.num_filters * 4, self.num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.avgpool1(x) 

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
