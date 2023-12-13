# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


class RRB(nn.Module):
    """Residual Refine Block"""
    def __init__(self, in_channels, out_channels=64):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


class SPPM(nn.Module):
    """Simple Pyramid Pooling Module"""
    def __init__(self, in_channels, out_channels):
        super(SPPM, self).__init__()
        self.sequential1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)
        )
        self.sequential2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(2),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)
        )
        self.sequential3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(4),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)
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


class NLDNN(nn.Module):
    """NLDNN for TF binding prediction"""
    def __init__(self, dim=4, motiflen=20):
        super(FCN, self).__init__()
        print("We are using NLDNN.")
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
        # general functions
        self.relu = nn.ELU(alpha=0.1)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
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

        return out_dense

################################
# The basic architectures in the dual-path framework
################################


class Generator(nn.Module):
    """Generator in the dual-path framework of adversarial training"""
    def __init__(self, dim=4, motiflen=20):
        super(Generator, self).__init__()
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
        # RRB
        self.RRB4 = RRB(64, 64)
        self.RRB3 = RRB(64, 64)
        self.RRB2 = RRB(64, 64)
        # general functions
        self.relu = nn.ELU(alpha=0.1)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def forward(self, data):
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
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
        out_dense = self.blend2(up2)

        return out_dense


class Predictor(nn.Module):
    """Predictor in the dual-path framework of adversarial training"""
    def __init__(self, size):
        super(Predictor, self).__init__()
        self.blend1 = bn_relu_conv(64, 1, kernel_size=3)
        self.size = size
        self._init_weights()

    def forward(self, data):
        up = upsample(data, self.size)
        out_dense = self.blend1(up)

        return out_dense


class Discriminator(nn.Module):
    """Discriminator in the dual-path framework of adversarial training"""
    def __init__(self, in_channels=64):
        super(Discriminator, self).__init__()
        #
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # classifier head
        c_in = 960
        self.bn = nn.BatchNorm1d(c_in)
        self.linear1 = nn.Linear(c_in, 64)
        self.drop = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(64, 1)
        # general functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        b, _, _ = data.size()
        # encode process
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)

        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)

        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        # classifier
        out2 = skip4.view(b, -1)
        out2 = self.bn(out2)
        out2 = self.linear1(out2)
        out2 = self.relu(out2)
        out2 = self.drop(out2)
        out2 = self.linear2(out2)

        return out2

################################
# The competing methods
################################


class FCNsignal(nn.Module):
    """FCNsignal for TF binding prediction"""
    def __init__(self, dim=4, motiflen=20):
        super(FCNsignal, self).__init__()
        print("We are using FCNsignal.")
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
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend1 = bn_relu_conv(64, 1, kernel_size=3)
        # general functions
        self.relu = nn.ELU(alpha=0.1, inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        out1 = out1.permute(0, 2, 1)
        out1_1, _ = self.gru1(out1)
        out1_2, _ = self.gru2(torch.flip(out1, [1]))
        out1 = out1_1 + out1_2
        out1 = self.gru_drop(out1)
        skip4 = out1.permute(0, 2, 1)
        up5 = self.aap(skip4)
        # decode process
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        out_dense = self.blend1(up1)

        return out_dense


class FCNA(nn.Module):
    """FCNA for TF binding prediction"""
    def __init__(self, dim=4, motiflen=20):
        super(FCNA, self).__init__()
        print("We are using FCNA.")
        # encode process
        self.conv1 = nn.Conv1d(in_channels=dim, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.aap = nn.AdaptiveAvgPool1d(1)
        # decode process
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # general functions
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        up5 = self.aap(out1)
        # decode process
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        out_dense = self.blend1(up1)

        return out_dense


class BPNet(nn.Module):
    """building BPNet on the Pytorch platform for TF binding prediction."""
    def __init__(self, dim=4, motiflen=25, batchnormalization=True):
        super(BPNet, self).__init__()
        print("We are using BPNet.")
        self.conv1 = nn.Conv1d(in_channels=dim, out_channels=64, kernel_size=motiflen, padding=motiflen // 2)
        self.relu1 = nn.ReLU(inplace=True)
        # sequential model
        self.sequential_model = nn.ModuleList()
        for i in range(1, 10):
            if batchnormalization:
                self.sequential_model.append((nn.Sequential(
                    nn.BatchNorm1d(64),
                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2**i, dilation=2**i),
                    nn.ReLU(inplace=True))))
            else:
                self.sequential_model.append((nn.Sequential(
                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2**i, dilation=2**i),
                    nn.ReLU(inplace=True))))
        self.convtranspose1 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=motiflen, padding=motiflen // 2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        b, c, l = data.size()
        x = self.conv1(data)
        x = self.relu1(x)
        for module in self.sequential_model:
            conv_x = module(x)
            x = conv_x + x
        bottleneck = x
        out = self.convtranspose1(bottleneck)

        return out

        
class Leopard(nn.Module):
    """building Leopard on the Pytorch platform for TF binding prediction."""
    def __init__(self, dim=4):
        super(Leopard, self).__init__()
        print("We are using Leopard.")
        num_blocks = 5
        self.num_blocks = num_blocks
        initial_filter = 15
        scale_filter = 1.5
        size_kernel = 7
        # activation = 'relu'
        # padding = 'same'
        layer_down = []
        layer_up = []
        layer_conv = []
        conv0 = nn.Sequential(
            nn.Conv1d(dim, initial_filter, kernel_size=size_kernel, padding=size_kernel//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(initial_filter),
            nn.Conv1d(initial_filter, initial_filter, kernel_size=size_kernel, padding=size_kernel // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(initial_filter)
        )
        layer_down.append(conv0)
        in_channel = initial_filter

        for i in range(num_blocks):
            out_channel = int(in_channel * scale_filter)
            layer = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channel, out_channel, kernel_size=size_kernel, padding=size_kernel // 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channel),
                nn.Conv1d(out_channel, out_channel, kernel_size=size_kernel, padding=size_kernel // 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channel)
            )
            in_channel = out_channel
            layer_down.append(layer)
        self.layer_down = nn.ModuleList(layer_down)

        for i in range(num_blocks):
            out_channel = round(in_channel / scale_filter)
            if i < 2:
                layer = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=3, stride=2)
            else:
                layer = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2)
            layer_up.append(layer)
            layer = nn.Sequential(
                nn.Conv1d(out_channel*2, out_channel, kernel_size=size_kernel, padding=size_kernel // 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channel),
                nn.Conv1d(out_channel, out_channel, kernel_size=size_kernel, padding=size_kernel // 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(out_channel)
            )
            layer_conv.append(layer)
            in_channel = out_channel
        self.layer_up = nn.ModuleList(layer_up)
        self.layer_conv = nn.ModuleList(layer_conv)
        self.outlayer = nn.Conv1d(in_channel, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        b, c, l = data.size()
        x = data
        feature_down = []
        for layer in self.layer_down:
            x = layer(x)
            feature_down.append(x)
        y = x
        for i in range(self.num_blocks):
            y = torch.cat((self.layer_up[i](y), feature_down[-(i+2)]), dim=1)
            y = self.layer_conv[i](y)
        out = self.outlayer(y)

        return out


class Transfer(nn.Module):
    """building transfer learning model for cross-species prediction."""
    def __init__(self, in_channels=64):
        super(Transfer, self).__init__()
        self.aap = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.linear2 = nn.Linear(in_channels, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        b, _, _ = x.size()
        out_dense = self.aap(x)
        out_dense = out_dense.view(b, -1)
        out_dense = self.linear1(out_dense)
        out_dense = self.relu(out_dense)
        out_dense = self.drop(out_dense)
        out_dense = self.linear2(out_dense)
        out_dense = self.sigmoid(out_dense)

        return out_dense
