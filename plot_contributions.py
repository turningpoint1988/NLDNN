#!/usr/bin/python

import os
import sys
import argparse
import h5py
import numpy as np
import os.path as osp
from Bio import SeqIO
import pyBigWig
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import DeepLift, IntegratedGradients
# custom functions defined by user
from model import NLDNN
from model_contrib import NLDNN_Contrib
import params
from datasets import SourceDataSet, TargetDataSet
import viz_sequence


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


# save data in hdf5 format
def outputHDF5(data, signal, peak, bed, filename):
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=data, **comp_kwargs)
        f.create_dataset('signal', data=signal, **comp_kwargs)
        f.create_dataset('peak', data=peak, **comp_kwargs)
        f.create_dataset('bed', data=bed, **comp_kwargs)


def encode(keys, windows_dict, outfile, sequence_dict, signal_file):
    data_all = []
    signal_all = []
    peak_all = []
    bed_all = []
    for key in keys:
        key_split = key.split('-')
        chrom = key_split[0]
        start = int(key_split[1])
        end = int(key_split[2])
        if end -start < WINDOW:
            continue
        bed_all.append([start, end])
        seq = str(sequence_dict[chrom].seq[start:end])
        data = one_hot(seq)
        data_all.append(data)
        signal = getbigwig(signal_file, chrom, start, end)
        signal[np.isnan(signal)] = 0.
        signal_all.append(signal)
        beds = windows_dict[key]
        peak = np.zeros(WINDOW)
        for bed in beds:
            bed_split = bed.split('-')
            start_o = int(bed_split[1]) - start
            end_o = int(bed_split[2]) - start
            peak[start_o:end_o] = 1
        peak_all.append(peak)

    data_all = np.array(data_all, dtype=np.float32)
    data_all = data_all.transpose((0, 2, 1))
    # signal_all = [[x] for x in signal_all]
    signal_all = np.array(signal_all, dtype=np.float32)
    signal_all = np.log10(1 + signal_all)
    peak_all = np.array(peak_all, dtype=np.float32)
    bed_all = np.array(bed_all, dtype=np.uint32)
    outputHDF5(data_all, signal_all, peak_all, bed_all, outfile)


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


def get_coordinate(peak):
    peak_all = []
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
            peak_all.append([start, end])

    return peak_all


def lineplot(df, peak, bed, out_f):
    sns.set_theme(style="white")
    fig, ax = plt.subplots()
    sns.despine(fig)
    g = sns.relplot(data=df, x="x", y="value", row="type", hue="type",
                    kind="line", linewidth=1, palette="crest", legend=False,
                    height=2, aspect=10)
    ##
    ax1 = g.axes_dict['Coverage']
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
    ax2 = g.axes_dict['NLDNN']
    ax2.set_xmargin(0)
    ax2.set_xlabel("chr1:{}-{}".format(bed[0], bed[1]), fontsize=12)
    ax2.set_ylabel("NLDNN", fontsize=12)
    #
    g.set_titles("")
    plt.savefig(out_f, format='png', bbox_inches='tight', dpi=300)


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Plot contributions and ISSM values for each DNA sequence (100kb)")

    parser.add_argument("-r", dest="root", type=str, default=None,
                        help="The path of the project.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of one of Cell.TF.")
    parser.add_argument("-m", dest="model", type=str, default='models',
                        help="The name of adopted method.")

    return parser.parse_args()


def main():
    args = get_args()
    device = params.device
    name = args.name
    if not osp.exists(osp.join(args.root, 'chr1/100kb/{}'.format(name))):
        os.makedirs(osp.join(args.root, 'chr1/100kb/{}'.format(args.name)))
    if not osp.exists(osp.join(args.root, 'chr1/100kb/{}/Human.chr1.hdf5'.format(name))):
        chrom_size_file = osp.join(args.root, 'Genome/hg38.chrom1.size')
        windows_out = osp.join(args.root, 'chr1/100kb/{}/Human_chr1_100kb.bed'.format(name))
        os.system('bedtools makewindows -g {} -w {} -s {} >'
                  ' {}'.format(chrom_size_file, WINDOW, WINDOW, windows_out))
        chipseq_file = osp.join(args.root, 'Human-Mouse/{}/ChIPseq.Human.{}.idr.bed'.format(name, name))
        signal_file = osp.join(args.root, 'Human-Mouse/{}/ChIPseq.Human.{}.pv.bigWig'.format(name, name))
        overlap_out = osp.join(args.root, 'chr1/100kb/{}/Human_chr1_100kb_overlap.bed'.format(name))
        os.system('bedtools intersect -wa -wb -a {} -b {} > {}'.format(windows_out, chipseq_file, overlap_out))
        keys, windows_dict = readfile(overlap_out)
        outfile = osp.join(args.root, 'chr1/100kb/{}/Human.chr1.hdf5'.format(name))
        genomefile = args.root + '/Genome/hg38.fa'
        sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genomefile), 'fasta'))
        encode(keys, windows_dict, outfile, sequence_dict, signal_file)
    if not osp.exists(osp.join(args.root, 'chr1/100kb/{}/Mouse.chr1.hdf5'.format(name))):
        chrom_size_file = osp.join(args.root, 'Genome/mm10.chrom1.size')
        windows_out = osp.join(args.root, 'chr1/100kb/{}/Mouse_chr1_100kb.bed'.format(name))
        os.system('bedtools makewindows -g {} -w {} -s {} >'
                  ' {}'.format(chrom_size_file, WINDOW, WINDOW, windows_out))
        chipseq_file = osp.join(args.root, 'Human-Mouse/{}/ChIPseq.Mouse.{}.idr.bed'.format(name, name))
        signal_file = osp.join(args.root, 'Human-Mouse/{}/ChIPseq.Mouse.{}.pv.bigWig'.format(name, name))
        overlap_out = osp.join(args.root, 'chr1/100kb/{}/Mouse_chr1_100kb_overlap.bed'.format(name))
        os.system('bedtools intersect -wa -wb -a {} -b {} > {}'.format(windows_out, chipseq_file, overlap_out))
        keys, windows_dict = readfile(overlap_out)
        outfile = osp.join(args.root, 'chr1/100kb/{}/Mouse.chr1.hdf5'.format(name))
        genomefile = args.root + '/Genome/mm10.fa'
        sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genomefile), 'fasta'))
        encode(keys, windows_dict, outfile, sequence_dict, signal_file)
    # # source for source
    with h5py.File(osp.join(args.root, 'chr1/100kb/{}/{}.chr1.hdf5'.format(name, params.src_dataset)), 'r') as f:
        Data = np.array(f['data'], dtype=np.float32)
        signal_t = np.array(f['signal'], dtype=np.float32)
        peak = np.array(f['peak'], dtype=np.float32)
        bed = np.array(f['bed'], dtype=np.float32)

    test_loader = DataLoader(TargetDataSet(Data), batch_size=1, shuffle=False, num_workers=0)
    # Load weights
    src_file = osp.join(args.root, args.model, args.name, '{}.model.best.pth'.format(params.src_dataset))
    chk = torch.load(src_file)
    state_dict = chk['model_state_dict']
    model = NLDNN(dim=4, motiflen=20)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    #
    print("{}: {} for {}".format(args.name, params.src_dataset, params.src_dataset))
    signal_p = []
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(params.device)
        with torch.no_grad():
            pred = model(X_data)
        pred = pred.view(-1).data.cpu().numpy()
        signal_p.append(pred)
    signal_p = np.array(signal_p)
    # # contribution for each peak
    device = torch.device("cpu")
    src_file = osp.join(args.root, args.model, args.name, '{}.model.best.pth'.format(params.src_dataset))
    chk = torch.load(src_file)
    state_dict = chk['model_state_dict']
    model_contrib = NLDNN_Contrib(dim=4, motiflen=20)
    model_contrib.load_state_dict(state_dict)
    model_contrib.to(device)
    model_contrib.eval()
    dl = DeepLift(model_contrib)
    index = 0 # appoint a DNA region
    for d, pred, true, pk, b in zip(Data[index], signal_p[index], signal_t[index], peak[index], bed[index]):
        #
        coord_pk = get_coordinate(pk)
        #
        out_f = osp.join(args.root, 'chr1/100kb/{}/{}.png'.format(name, index))
        type = ['Coverage'] * len(true) + ['NLDNN'] * len(pred)
        x = list(range(WINDOW)) + list(range(WINDOW))
        value = np.concatenate((true, pred))
        assert len(pred) == len(list(range(WINDOW))) and len(true) == len(list(range(WINDOW))), "not consistent"
        df = pd.DataFrame({'x': x, 'value': value, 'type': type})
        lineplot(df, pk, b, out_f)
        # computing contribution and issm for each peak
        if not osp.exists(osp.join(args.root, 'chr1/100kb/{}/index{}'.format(name, index))):
            os.makedirs(osp.join(args.root, 'chr1/100kb/{}/index{}'.format(name, index)))
        count = 0
        for coord in coord_pk:
            start, end = relocation(coord[0], coord[1], 200)
            data = d[:, start:end]
            data_one = np.array([data], dtype=np.float32)  # 1x4xL
            data_one = torch.from_numpy(data_one)
            data_one.requires_grad = True
            ref_one = torch.zeros_like(data_one)
            #
            model_contrib.zero_grad()
            contribution = dl.attribute(data_one, ref_one, target=0, return_convergence_delta=False)
            contribution = contribution.data.numpy()
            contribution[np.isnan(contribution)] = 0
            contribution = contribution[0]
            contribution = contribution.transpose((1, 0))
            #
            count += 1
            viz_sequence.plot_weights(contribution, subticks_frequency=20, figsize=(20, 2),
                                      out_f=osp.join(args.root,
                                                     'chr1/100kb/{}/index{}/peak{}.png'.format(name, index, count))
                                      )
            #
            out_f = osp.join(args.root, 'chr1/100kb/{}/index{}/peak_in_silo{}.png'.format(name, index, count))
            in_silo(data, model, out_f)


if __name__ == "__main__":
    main()

