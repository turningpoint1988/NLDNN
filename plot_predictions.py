#!/usr/bin/python

import os
import sys
import argparse
import h5py
import numpy as np
from copy import deepcopy
import os.path as osp
from Bio import SeqIO
import pyBigWig
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# custom functions defined by user
from model import NLDNN, Generator, Predictor
from datasets import SourceDataSet, TargetDataSet

WINDOW = 100000


def lineplot(df, peak, bed, out_f):
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots()
    sns.despine(fig)
    g = sns.relplot(data=df, x="x", y="value", row="type", hue="type",
                    kind="line", linewidth=1, palette="crest", legend=False,
                    height=2, aspect=10)
    ##
    ax1 = g.axes_dict['True']
    i = 0
    while i < len(peak):
        if peak[i] == 0:
            i += 1
        else:
            start = i
            j = start
            while j < len(peak):
                if peak[j] == 0:
                    end = j - 1
                    i = j
                    break
                j += 1
            if j == len(peak):
                end = j - 1
                i = j
            ax1.hlines(y=-0.1, xmin=start-100, xmax=end+100, colors='red', linewidth=2)
    ##
    ax1.set_xmargin(0)
    ax1.set_ylabel("Coverage", fontsize=12)
    #
    ax2 = g.axes_dict['Human']
    ax2.set_xmargin(0)
    ax2.set_ylabel("Human Model", fontsize=12)
    #
    ax3 = g.axes_dict['Mouse']
    ax3.set_xmargin(0)
    ax3.set_ylabel("Mouse Model", fontsize=12)
    #
    ax4 = g.axes_dict['Adaptation']
    ax4.set_xmargin(0)
    ax4.set_xlabel("chr1:{}-{}".format(bed[0], bed[1]), fontsize=12)
    ax4.set_ylabel("Mouse Adaptation", fontsize=12)
    #
    g.set_titles("")
    plt.savefig(out_f, format='png', bbox_inches='tight', dpi=300)


def one_hot(seq):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


def getbigwig(file, chrom, start, end):
    bw = pyBigWig.open(file)
    sample = np.array(bw.values(chrom, start, end))
    bw.close()
    return sample


def relocation(start, end, seq_len, window=100000):
    original_len = end - start
    if original_len < seq_len:
        start_update = start - np.ceil((seq_len - original_len) / 2)
    elif original_len > seq_len:
        start_update = start + np.ceil((original_len - seq_len) / 2)
    else:
        start_update = start
    if start_update < 0:
        start_update = 0
    end_update = start_update + seq_len
    if end_update > window:
        end_update = window
        start_update = end_update - seq_len

    return int(start_update), int(end_update)


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Plot predictions from human, mosue, mouse-adaptation models "
                                                 "for each DNA sequence (100kb)")

    parser.add_argument("-r", dest="root", type=str, default=None,
                        help="The path of the project.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of one of Cell.TF.")
    parser.add_argument("-m", dest="model", type=str, default='models',
                        help="The name of adopted method.")

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda:0')
    src_dataset = 'Human'
    tgt_dataset = 'Mouse'
    with h5py.File(osp.join(args.root, 'chr1/100kb/{}/{}.chr1.hdf5'.format(args.name, src_dataset)), 'r') as f:
        Data = np.array(f['data'], dtype=np.float32)
        signal_t = np.array(f['signal'], dtype=np.float32)
        peak = np.array(f['peak'], dtype=np.float32)
        bed = np.array(f['bed'], dtype=np.uint32)

    test_loader = DataLoader(TargetDataSet(Data), batch_size=1, shuffle=False, num_workers=0)
    # # source for source
    print("{}: {} for {}".format(args.name, src_dataset, src_dataset))
    # Load weights
    src_file = osp.join(args.root, args.model, args.name, '{}.model.best.pth'.format(src_dataset))
    chk = torch.load(src_file)
    state_dict = chk['model_state_dict']
    model = NLDNN(dim=4, motiflen=20)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    signal_ps = []
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            pred = model(X_data)
        pred = pred.view(-1).data.cpu().numpy()
        signal_ps.append(pred)
    signal_ps = np.array(signal_ps)
    #
    print("{}: {} for {}".format(args.name, tgt_dataset, src_dataset))
    # Load weights
    tgt_file = osp.join(args.root, args.model, args.name, '{}.model.best.pth'.format(tgt_dataset))
    chk = torch.load(tgt_file)
    state_dict = chk['model_state_dict']
    model = NLDNN(dim=4, motiflen=20)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    #
    signal_pt = []
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            pred = model(X_data)
        pred = pred.view(-1).data.cpu().numpy()
        signal_pt.append(pred)
    signal_pt = np.array(signal_pt)
    #
    print("{}: {} Adaptation for {}".format(args.name, tgt_dataset, src_dataset))
    # # Load weights
    tgt_generator_file = osp.join(args.root, args.model, args.name, '{}.tgt.generator.pth'.format(tgt_dataset))
    chk = torch.load(tgt_generator_file)
    state_dict = chk['model_state_dict']
    tgt_generator = Generator(dim=4, motiflen=20)
    tgt_generator.load_state_dict(state_dict)
    tgt_generator.to(device)
    tgt_generator.eval()
    #
    src_predictor_file = osp.join(args.root, args.model, args.name, '{}.src.predictor.pth'.format(tgt_dataset))
    chk = torch.load(src_predictor_file)
    state_dict = chk['model_state_dict']
    src_predictor = Predictor(size=WINDOW)
    src_predictor.load_state_dict(state_dict)
    src_predictor.to(device)
    src_predictor.eval()
    #
    signal_at = []
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            pred = src_predictor(tgt_generator(X_data))
        pred = pred.view(-1).data.cpu().numpy()
        signal_at.append(pred)
    signal_at = np.array(signal_at)
    # plot
    index = 0  # appoint a DNA region (100kb)
    for t, ps, pt, at, pk, b in zip(signal_t, signal_ps, signal_pt, signal_at, peak, bed):
        out_f = osp.join(args.root, 'chr1/100kb/{}/cross_{}.png'.format(args.name, index))
        type = ['True'] * len(t) + ['Human'] * len(ps) + ['Mouse'] * len(pt) + ['Adaptation'] * len(at)
        x = list(range(WINDOW)) * 4
        value = np.concatenate((t, ps, pt, at))
        df = pd.DataFrame({'x': x, 'value': value, 'type': type})
        lineplot(df, pk, b, out_f)


if __name__ == "__main__":
    main()

