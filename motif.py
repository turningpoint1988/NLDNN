#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import os.path as osp
import params
import h5py

import torch
from torch.utils.data import DataLoader

# custom functions defined by user
from datasets import SourceDataSet
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
    def __init__(self, dim=4, motiflen=20):
        super(NLDNN, self).__init__()
        print("We are using NLDNN for motif discovery.")
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
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        score = out1
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

        return out_dense[0], score[0]
        

def extract(signal, thres):
    start = end = 0
    seqLen = len(signal)
    position = np.argmax(signal)
    Max = np.max(signal)
    if Max > thres:
        start = position - WINDOW // 2
        end = position + WINDOW // 2
        if start < 0: start = 0
        if end > seqLen - 1:
            end = seqLen - 1
            start = end - WINDOW

    return int(start), int(end)


def motif_all(model, test_loader, outdir, thres=0.5):
    # for test data
    count = 0
    motif_data = [0.] * kernel_num
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(params.device)
        with torch.no_grad():
            signal_p, score_p = model(X_data)
        signal_p = signal_p.view(-1).data.cpu().numpy()
        start, end = extract(signal_p, thres)
        if start == end:
            continue
        count += 1
        data = X_data[0].data.cpu().numpy()
        score_p = score_p.data.cpu().numpy()
        score_p = score_p[:, start:end]
        max_index = np.argmax(score_p, axis=1)
        for i in range(kernel_num):
            index = max_index[i]
            index += start
            data_slice = data[:, index:(index + motifLen)]
            motif_data[i] += data_slice
    print("The number of selected samples is {}".format(count))
    pfms = compute_pfm(motif_data)
    writeFile(pfms, 'motif', outdir)


def compute_pfm(motifs):
    pfms = []
    informations = []
    for motif in motifs:
        if np.sum(motif) == 0.: continue
        sum_ = np.sum(motif, axis=0)
        if sum_[0] < 10: continue
        pfm = motif / sum_
        pfms.append(pfm)
        #
        row, col = pfm.shape
        information = 0
        for j in range(col):
            information += 2 + np.sum(pfm[:, j] * np.log2(pfm[:, j]+1e-8))
        informations.append(information)
    pfms_filter = []
    index = np.argsort(np.array(informations))
    index = index[::-1]
    for i in range(len(informations)):
        index_c = index[i]
        pfms_filter.append(pfms[index_c])
    return pfms_filter


def writeFile(pfm, flag, outdir):
    out_f = open(outdir + '/{}.meme'.format(flag), 'w')
    out_f.write("MEME version 5.3.3\n\n")
    out_f.write("ALPHABET= ACGT\n\n")
    out_f.write("strands: + -\n\n")
    out_f.write("Background letter frequencies\n")
    out_f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
    for i in range(len(pfm)):
        out_f.write("MOTIF " + "{}\n".format(i+1))
        out_f.write("letter-probability matrix: alength= 4 w= {} nsites= {}\n".format(motifLen, motifLen))
        current_pfm = pfm[i]
        for col in range(current_pfm.shape[1]):
            for row in range(current_pfm.shape[0]):
                out_f.write("{:.4f} ".format(current_pfm[row, col]))
            out_f.write("\n")
        out_f.write("\n")
    out_f.close()


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model itself for motif location")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="The directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of one of Cell.TF.")
    parser.add_argument("-dim", dest="dimension", type=int, default=4,
                        help="The channel number of input.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    parser.add_argument("-t", dest="thres", type=float, default=0.5,
                        help="A threshold value.")
    parser.add_argument("-o", dest="outdir", type=str, default='./motifs/',
                        help="Where to save experimental results.")

    return parser.parse_args()


args = get_args()
dim = args.dimension
motifLen = 20
WINDOW = 61
kernel_num = 64


def main():
    name = args.name
    with h5py.File(osp.join(args.data_dir, '{}.{}.te.pos.hdf5'.format(params.src_dataset, name)), 'r') as f:
        data_te_pos = np.array(f['data'], dtype=np.float32)
        label_te_pos = np.array(f['signal'], dtype=np.float32)
    #
    data_te_pos_loader = DataLoader(SourceDataSet(data_te_pos, label_te_pos), batch_size=1,
                                    shuffle=False, num_workers=0)
    # Load weights
    checkpoint_file = osp.join(args.checkpoint, '{}.model.best.pth'.format(params.src_dataset))
    chk = torch.load(checkpoint_file, map_location='cuda:0')
    state_dict = chk['model_state_dict']
    model = NLDNN(dim=dim, motiflen=motifLen)
    model.load_state_dict(state_dict)
    model.to(params.device)
    model.eval()
    motif_all(model, data_te_pos_loader, args.outdir, args.thres)


if __name__ == "__main__":
    main()

