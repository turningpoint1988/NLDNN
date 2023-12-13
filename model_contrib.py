#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


class RRB(nn.Module):
    """Residual Refine Block"""
    def __init__(self, in_channels, out_channels=64):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = nn.ReLU()(res)
        res = self.conv3(res)
        return nn.ReLU()(x + res)


class SPPM(nn.Module):
    """Simple Pyramid Pooling Module"""
    def __init__(self, in_channels, out_channels):
        super(SPPM, self).__init__()
        self.sequential1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU()
        )
        self.sequential2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(2),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU()
        )
        self.sequential3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(4),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU()
        )
        self.conv_out = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU()
        )

    def forward(self, x):
        size = x.size()[-1]
        x1 = self.sequential1(x)
        up1 = upsample(x1, size)
        x2 = self.sequential2(x)
        up2 = upsample(x2, size)
        x3 = self.sequential3(x)
        up3 = upsample(x3, size)

        fusion = up1 + up2 + up3
        out = self.conv_out(fusion)

        return out


class NLDNN_Contrib(nn.Module):
    def __init__(self, dim=4, motiflen=20):
        super(NLDNN_Contrib, self).__init__()
        print("We are using NLDNN for computing contributions.")
        # encode process
        self.conv1 = nn.Conv1d(in_channels=dim, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # decode process
        self.gru1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru_drop = nn.Dropout(p=0.2)
        self.sppm = SPPM(64, 64)
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend1 = bn_relu_conv(64, 1, kernel_size=3)
        # RRB
        self.RRB4 = RRB(64, 64)
        self.RRB3 = RRB(64, 64)
        self.RRB2 = RRB(64, 64)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = nn.ReLU()(out1)
        out1 = self.pool1(out1)
        out1 = nn.Dropout(p=0.2)(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = nn.ReLU()(out1)
        out1 = self.pool2(out1)
        out1 = nn.Dropout(p=0.2)(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = nn.ReLU()(out1)
        out1 = self.pool3(out1)
        out1 = nn.Dropout(p=0.2)(out1)
        out1 = out1.permute(0, 2, 1)
        out1_1, _ = self.gru1(out1)
        out1_2, _ = self.gru2(torch.flip(out1, [1]))
        out1 = out1_1 + out1_2
        out1 = self.gru_drop(out1)
        skip4 = out1.permute(0, 2, 1)
        # decode process
        up4 = self.sppm(skip4)
        up4 = up4 + self.RRB4(skip4)
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + self.RRB3(skip3)
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + self.RRB2(skip2)
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        out_dense = self.blend1(up1)
        out_dense, _ = torch.max(out_dense, dim=2)

        return out_dense
