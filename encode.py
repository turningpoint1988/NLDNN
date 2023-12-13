# coding:utf-8
import os.path as osp
import os
import random
import numpy as np
from Bio import SeqIO
import pyBigWig
import h5py
import argparse

SEQ_LEN = 600
INDEX = ['chr' + str(i + 1) for i in range(23)]
INDEX[22] = 'chrX'


def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    seq = str(sequence_dict[chrom].seq[start:end])
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


def getbigwig(file, chrom, start, end):
    bw = pyBigWig.open(file)
    sample = np.array(bw.values(chrom, start, end))
    bw.close()
    return sample


# save data in hdf5 format
def outputHDF5(data, label, signal, filename, dataname='data', labelname='label', signalname='signal'):
    print('data shape: {}\tlable shape: {}\tsignal shape: {}\n'.format(data.shape, label.shape, signal.shape))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(filename, 'w') as f:
        f.create_dataset(dataname, data=data, **comp_kwargs)
        f.create_dataset(labelname, data=label, **comp_kwargs)
        f.create_dataset(signalname, data=signal, **comp_kwargs)


def datasplit(seq_dir, sequence_dict, signal_file, status, out_dir, name, upper=0):
    #
    if status == 'tr':
        pos_file = seq_dir + '/train_pos_shuf.bed'
    elif status == 'te':
        pos_file = seq_dir + '/test_pos_shuf.bed'
    elif status == 'va':
        pos_file = seq_dir + '/validation_pos_shuf.bed'
    else:
        pos_file = None
    data_pos = []
    label_pos = []
    signal_pos = []
    with open(pos_file) as f:
        for line in f:
            line_split = line.strip().split()
            chrom = line_split[0]
            start = int(line_split[1])
            end = int(line_split[2])
            label = int(line_split[3])
            data = one_hot(sequence_dict, chrom, start, end)
            signal = getbigwig(signal_file, chrom, start, end)
            signal[np.isnan(signal)] = 0.
            data_pos.append(data)
            signal_pos.append(signal)
            label_pos.append(label)

        data_pos = np.array(data_pos, dtype=np.float32)
        data_pos = data_pos.transpose((0, 2, 1))
        signal_pos = [[x] for x in signal_pos]
        signal_pos = np.array(signal_pos, dtype=np.float32)
        signal_pos = np.log10(1 + signal_pos)
        label_pos = [[x] for x in label_pos]
        label_pos = np.array(label_pos, dtype=np.float32)
        out_f = out_dir + '/{}.{}.pos.hdf5'.format(name, status)
        outputHDF5(data_pos, label_pos, signal_pos, out_f)
    # save negative data in batches
    if upper > 0:
        number_pos = data_pos.shape[0] * 3
    else:
        number_pos = data_pos.shape[0] * 10

    if status == 'tr':
        neg_file = seq_dir + '/train_neg_shuf.bed'
    elif status == 'te':
        neg_file = seq_dir + '/test_neg_shuf.bed'
    elif status == 'va':
        neg_file = seq_dir + '/validation_neg_shuf.bed'
    else:
        neg_file = None
    data_neg = []
    label_neg = []
    signal_neg = []
    count = 0
    flag = 0
    with open(neg_file) as f:
        for line in f:
            line_split = line.strip().split()
            chrom = line_split[0]
            start = int(line_split[1])
            end = int(line_split[2])
            label = int(line_split[3])
            data = one_hot(sequence_dict, chrom, start, end)
            signal = np.zeros(SEQ_LEN)
            data_neg.append(data)
            signal_neg.append(signal)
            label_neg.append(label)
            count += 1
            if count == number_pos:
                data_neg = np.array(data_neg, dtype=np.float32)
                data_neg = data_neg.transpose((0, 2, 1))
                signal_neg = [[x] for x in signal_neg]
                signal_neg = np.array(signal_neg, dtype=np.float32)
                label_neg = [[x] for x in label_neg]
                label_neg = np.array(label_neg, dtype=np.float32)
                out_f = out_dir + '/{}.{}.neg{}.hdf5'.format(name, status, flag)
                outputHDF5(data_neg, label_neg, signal_neg, out_f)
                flag += 1
                #
                data_neg = []
                label_neg = []
                signal_neg = []
                count = 0
            if 0 < upper <= flag:
                return
        if count > 0:
            data_neg = np.array(data_neg, dtype=np.float32)
            data_neg = data_neg.transpose((0, 2, 1))
            signal_neg = [[x] for x in signal_neg]
            signal_neg = np.array(signal_neg, dtype=np.float32)
            label_neg = [[x] for x in label_neg]
            label_neg = np.array(label_neg, dtype=np.float32)
            out_f = out_dir + '/{}.{}.neg{}.hdf5'.format(name, status, flag)
            outputHDF5(data_neg, label_neg, signal_neg, out_f)


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Encoding DNA sequences into one-hot matrices")

    parser.add_argument("-r", dest="root", type=str, default=None,
                        help="The root of this project.")
    parser.add_argument("-t", dest="target", type=str, default=None,
                        help="The name of one of Cell.TF.")

    return parser.parse_args()


def main():
    random.seed(666)
    args = get_args()
    ROOT = args.root
    target = args.target
    genome = ROOT + '/Genome'
    data_dir = ROOT + '/Human-Mouse/{}'.format(target)
    out_dir = osp.join(data_dir, 'data')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    # Human
    seq_dir = data_dir + '/raw_data/hg38'
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome + '/hg38.fa'), 'fasta'))
    signal_file = data_dir + '/ChIPseq.Human.{}.pv.bigWig'.format(target)
    datasplit(seq_dir, sequence_dict, signal_file, 'tr', out_dir, 'Human.'+target, 60)
    datasplit(seq_dir, sequence_dict, signal_file, 'va', out_dir, 'Human.'+target, 10)
    datasplit(seq_dir, sequence_dict, signal_file, 'te', out_dir, 'Human.'+target, 0)
    # # Mouse
    seq_dir = data_dir + '/raw_data/mm10'
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome + '/mm10.fa'), 'fasta'))
    signal_file = data_dir + '/ChIPseq.Mouse.{}.pv.bigWig'.format(target)
    datasplit(seq_dir, sequence_dict, signal_file, 'tr', out_dir, 'Mouse.'+target, 10)
    datasplit(seq_dir, sequence_dict, signal_file, 'va', out_dir, 'Mouse.'+target, 1)
    datasplit(seq_dir, sequence_dict, signal_file, 'te', out_dir, 'Mouse.' + target, 0)


if __name__ == '__main__':  main()
