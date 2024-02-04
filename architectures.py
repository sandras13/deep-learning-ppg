import torch
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class DeepConvLSTM(nn.Module):
    def __init__(self, num_channels, num_classes, num_layers, hidden_size):
        super(DeepConvLSTM, self).__init__()
        self.num_filters = 64
        self.num_convblocks = 4

        self.conv1 = ConvBlock(num_channels, self.num_filters, kernel_size=7, stride=2, padding=3, use_relu=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool1d(kernel_size=4, stride=4)

        self.convblocks = nn.ModuleList()
        for i in range(self.num_convblocks):
            self.convblocks.append(ConvBlock(self.num_filters, self.num_filters, kernel_size=3, stride=1, padding=1, use_relu=True))

        self.lstm = nn.LSTM(input_size=self.num_filters, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
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
    def __init__(self, in_channels, out_channels, opt, stride=1):
        super(ResidualBlock, self).__init__()
        self.opt = opt
        # 0 : single-branch ResBlock
        # 1 : 1-3-5 ResBlock
        # 2 : 3-5-7 ResBlock
        # 3 : 3-5 ResBlock
        # 4 : 1-3-5 ResBlock with Bottleneck

        if self.opt == 0:
            self.conv1_3 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2_3 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_relu=False)

        elif self.opt < 4:
            self.conv1_3 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2_3 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_relu=False)

            self.conv1_5 = ConvBlock(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
            self.conv2_5 = ConvBlock(out_channels, out_channels, kernel_size=5, stride=1, padding=2, use_relu=False)

            if self.opt == 1:
                self.conv1_1 = ConvBlock(in_channels, out_channels, kernel_size=1, padding = 0, stride=stride)
                self.conv2_1 = ConvBlock(out_channels, out_channels, kernel_size=1, padding = 0, stride=1, use_relu=False)
            
            elif self.opt == 2:
                self.conv1_1 = ConvBlock(in_channels, out_channels, kernel_size=7, padding = 3, stride=stride)
                self.conv2_1 = ConvBlock(out_channels, out_channels, kernel_size=7, padding = 3, stride=1, use_relu=False)
                
        else:
            self.conv1_1 = ConvBlock(in_channels, out_channels, kernel_size=1, padding = 0, stride=stride, use_relu=False)

            self.conv0_3 = ConvBlock(in_channels, out_channels//2, kernel_size=1, stride=stride, padding=0)
            self.conv1_3 = ConvBlock(out_channels//2, out_channels//2, kernel_size=3, stride=stride, padding=1)
            self.conv2_3 = ConvBlock(out_channels//2, out_channels, kernel_size=1, stride=1, padding=0, use_relu=False)

            self.conv0_5 = ConvBlock(in_channels, out_channels//2, kernel_size=1, stride=stride, padding=0)
            self.conv1_5 = ConvBlock(out_channels//2, out_channels//2, kernel_size=5, stride=stride, padding=2)
            self.conv2_5 = ConvBlock(out_channels//2, out_channels, kernel_size=1, stride=1, padding=0, use_relu=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, use_relu=False)

    def forward(self, x):
        residual = self.shortcut(x)

        if self.opt == 0:
            out = self.conv1_3(x)
            out = self.conv2_3(out)
        
        elif self.opt < 4:
            out_3 = self.conv1_3(x)
            out_3 = self.conv2_3(out_3)

            out_5 = self.conv0_5(x)
            out_5 = self.conv1_5(out_5)

            if self.opt != 3:
                out_1 = self.conv1_1(x)
                out_1 = self.conv2_1(out_1)
                out = out_1 + out_3 + out_5

            else: out = out_3 + out_5

        else:
            out_1 = self.conv1_1(x)

            out_3 = self.conv0_3(x)
            out_3 = self.conv1_3(out_3)
            out_3 = self.conv2_3(out_3)

            out_5 = self.conv0_5(x)
            out_5 = self.conv1_5(out_5)
            out_5 = self.conv2_5(out_5)

            out = out_1 + out_3 + out_5

        out = out + residual
        out = nn.functional.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_channels, num_classes, num_resblocks, opt):
        super(ResNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_resblocks = num_resblocks
        self.opt = opt
        self.num_filters = 64

        self.conv1 = ConvBlock(self.num_channels, self.num_filters, kernel_size=7, stride=2, padding=3, use_relu=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool1 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.resblocks = nn.ModuleList()
        for i in range(self.num_resblocks):
            self.resblocks.append(ResidualBlock(self.num_filters, self.num_filters, self.opt))

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



