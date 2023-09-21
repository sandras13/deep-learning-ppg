import torch
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class DeepConvLSTM(nn.Module):
    def __init__(self, num_channels, num_classes,  num_layers, hidden_size):
        super(DeepConvLSTM, self).__init__()
        self.num_filters = 64
        self.num_convblocks = 4

        self.avgpool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.conv1 = ConvBlock(num_channels, self.num_filters, kernel_size=7, stride=2, padding=3, use_relu=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.convblocks = nn.ModuleList()
        for i in range(self.num_convblocks):
            self.convblocks.append(ConvBlock(self.num_filters, self.num_filters, kernel_size=3, stride=1, padding=1, use_relu=True))
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.avgpool(x)

        for convblock in self.convblocks:
            x = convblock(x)

        x = x.permute(0, 2, 1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)
        
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bottleneck=False):
        super(ResidualBlock, self).__init__()
        self.bottleneck = bottleneck
        if bottleneck:
            self.conv1 = ConvBlock(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
            self.conv2 = ConvBlock(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1)
            self.conv3 = ConvBlock(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, use_relu=False)
        else:
            self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_relu=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, use_relu=False)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        if self.bottleneck:
            out = self.conv3(out)

        out = out + residual
        out = nn.functional.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_channels, num_classes, num_resblocks):
        super(ResNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_resblocks = num_resblocks
        self.num_filters = 64

        self.conv1 = ConvBlock(self.num_channels, self.num_filters, kernel_size=7, stride=2, padding=3, use_relu=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool1 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.resblocks = nn.ModuleList()
        for i in range(self.num_resblocks):
            self.resblocks.append(ResidualBlock(self.num_filters, self.num_filters, bottleneck=False))

        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.num_filters, self.num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.avgpool1(x)

        for resblock in self.resblocks:
            x = resblock(x)

        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        return x