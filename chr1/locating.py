#!/usr/bin/python

import os
import sys
import argparse
import h5py
import numpy as np
import os.path as osp
from Bio import SeqIO
import torch
from torch.utils.data import DataLoader
# custom functions defined by user
sys.path.append("..")
from model import Generator, Predictor, Discriminator, NLDNN
import params
from datasets import SourceDataSet

SEQ_LEN = 600
WINDOW = 100


def one_hot(seq):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


# save data in hdf5 format
def outputHDF5(data, bed, filename, dataname='data', bedname='bed'):
    print('data shape: {}\n'.format(data.shape))
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(filename, 'w') as f:
        f.create_dataset(dataname, data=data, **comp_kwargs)
        f.create_dataset(bedname, data=bed, **comp_kwargs)


def encode(infile, outfile, sequence_dict):
    data_all = []
    bed_all = []
    with open(infile) as f:
        for line in f:
            line_split = line.strip().split()
            chrom = line_split[0]
            start = int(line_split[1])
            end = int(line_split[2])
            seq = str(sequence_dict[chrom].seq[start:end]).upper()
            if 'N' in seq:
                continue
            data = one_hot(seq)
            data_all.append(data)
            bed_all.append([start, end])

        data_all = np.array(data_all, dtype=np.float32)
        data_all = data_all.transpose((0, 2, 1))
        bed_all = np.array(bed_all)
        outputHDF5(data_all, bed_all, outfile)


def locating_adap(generator, predictor, test_loader, outdir, name):

    f1 = open(osp.join(outdir, '{}.neg.bed'.format(name)), 'w')
    f2 = open(osp.join(outdir, '{}.pos.bed'.format(name)), 'w')
    num_pos = 0
    num_neg = 0
    max_all = []
    index_all = []
    bed_all = []
    # for test data
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(params.device)
        X_bed = sample_batch["label"].data.numpy()
        with torch.no_grad():
            signal_p = predictor(generator(X_data))
        signal_p = signal_p.view(signal_p.size()[0], -1).data.cpu().numpy()
        if i_batch == 0:
            max_all = np.max(signal_p, axis=1)
            index_all = np.argmax(signal_p, axis=1)
            bed_all = X_bed
        else:
            max_all = np.concatenate((max_all, np.max(signal_p, axis=1)))
            index_all = np.concatenate((index_all, np.argmax(signal_p, axis=1)))
            bed_all = np.concatenate((bed_all, X_bed))

    thres = np.quantile(max_all, q=0.99)
    # thres = np.max(max_all) * 0.5
    print("The threshold is {:.5f}".format(thres))
    for max_, index_, bed_ in zip(max_all, index_all, bed_all):
        if max_ <= thres:
            # print("The sequence is predicted to be a negative sample.")
            num_neg += 1
            position = index_ + bed_[0]
            start = position - WINDOW // 2
            end = position + WINDOW // 2
            f1.write("{}\t{}\t{}\t{}\n".format('chr1', start, end, position))
        else:
            # print("The sequence is predicted to be a positive sample.")
            num_pos += 1
            position = index_ + bed_[0]
            start = position - WINDOW // 2
            end = position + WINDOW // 2
            f2.write("{}\t{}\t{}\t{}\n".format('chr1', start, end, position))

    print("The number of positive and negative samples is {} and {}\n".format(num_pos, num_neg))
    f1.close()
    f2.close()


def locating(model, test_loader, outdir, name):

    f1 = open(osp.join(outdir, '{}.neg.bed'.format(name)), 'w')
    f2 = open(osp.join(outdir, '{}.pos.bed'.format(name)), 'w')
    num_pos = 0
    num_neg = 0
    max_all = []
    index_all = []
    bed_all = []
    # for test data
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(params.device)
        X_bed = sample_batch["label"].data.numpy()
        with torch.no_grad():
            signal_p = model(X_data)
        signal_p = signal_p.view(signal_p.size()[0], -1).data.cpu().numpy()
        if i_batch == 0:
            max_all = np.max(signal_p, axis=1)
            index_all = np.argmax(signal_p, axis=1)
            bed_all = X_bed
        else:
            max_all = np.concatenate((max_all, np.max(signal_p, axis=1)))
            index_all = np.concatenate((index_all, np.argmax(signal_p, axis=1)))
            bed_all = np.concatenate((bed_all, X_bed))

    thres = np.quantile(max_all, q=0.99)
    # thres = np.max(max_all) * 0.5
    print("The threshold is {:.5f}".format(thres))
    for max_, index_, bed_ in zip(max_all, index_all, bed_all):
        if max_ <= thres:
            # print("The sequence is predicted to be a negative sample.")
            num_neg += 1
            position = index_ + bed_[0]
            start = position - WINDOW // 2
            end = position + WINDOW // 2
            f1.write("{}\t{}\t{}\t{}\n".format('chr1', start, end, position))
        else:
            # print("The sequence is predicted to be a positive sample.")
            num_pos += 1
            position = index_ + bed_[0]
            start = position - WINDOW // 2
            end = position + WINDOW // 2
            f2.write("{}\t{}\t{}\t{}\n".format('chr1', start, end, position))

    print("The number of positive and negative samples is {} and {}, respectively\n".format(num_pos, num_neg))
    f1.close()
    f2.close()


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="NLDNN for locating TF binding regions")

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
    if not osp.exists(osp.join(args.root, 'chr1/data')):
        os.mkdir(osp.join(args.root, 'chr1/data'))
    if not osp.exists(osp.join(args.root, 'chr1/data/Human.chr1.hdf5')):
        infile = osp.join(args.root, 'chr1/Human.chrom1.size.bins.filtered')
        outfile = osp.join(args.root, 'chr1/data/Human.chr1.hdf5')
        genomefile = args.root + '/Genome/hg38.fa'
        sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genomefile), 'fasta'))
        encode(infile, outfile, sequence_dict)
    if not osp.exists(osp.join(args.root, 'chr1/data/Mouse.chr1.hdf5')):
        infile = osp.join(args.root, 'chr1/Mouse.chrom1.size.bins.filtered')
        outfile = osp.join(args.root, 'chr1/data/Mouse.chr1.hdf5')
        genomefile = args.root + '/Genome/mm10.fa'
        sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genomefile), 'fasta'))
        encode(infile, outfile, sequence_dict)

    # # load source data
    with h5py.File(osp.join(args.root, 'chr1/data/{}.chr1.hdf5'.format(params.src_dataset)), 'r') as f:
        s_data = np.array(f['data'], dtype=np.float32)
        s_bed = np.array(f['bed'], dtype=np.int64)
    s_te = SourceDataSet(s_data, s_bed)
    s_loader = DataLoader(s_te, batch_size=1000, shuffle=False, num_workers=0)
    # Load weights
    model_file = osp.join(args.root, args.model, args.name, '{}.model.best.pth'.format(params.src_dataset))
    chk = torch.load(model_file, map_location='cuda:0')
    state_dict = chk['model_state_dict']
    model = NLDNN(dim=4, motiflen=20)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("{}: {} for {}".format(args.name, params.src_dataset, params.src_dataset))
    out_dir = osp.join(args.root, 'chr1/location')
    locating(model, s_loader, out_dir, args.name + '.' + params.src_dataset + '.ss')
    # # load target data
    with h5py.File(osp.join(args.root, 'chr1/data/{}.chr1.hdf5'.format(params.tgt_dataset)), 'r') as f:
        t_data = np.array(f['data'], dtype=np.float32)
        t_bed = np.array(f['bed'], dtype=np.int64)
    t_te = SourceDataSet(t_data, t_bed)
    t_loader = DataLoader(t_te, batch_size=500, shuffle=False, num_workers=0)
    print("{}: {} for {}".format(args.name, params.src_dataset, params.tgt_dataset))
    out_dir = osp.join(args.root, 'chr1/location')
    locating(model, t_loader, out_dir, args.name + '.' + params.src_dataset + '.st')
    # # source adaptation for target
    src_predictor_file = osp.join(args.root, args.model, args.name, '{}.src.predictor.pth'.format(params.src_dataset))
    chk = torch.load(src_predictor_file)
    state_dict = chk['model_state_dict']
    src_predictor = Predictor(size=SEQ_LEN)
    src_predictor.load_state_dict(state_dict)
    src_predictor.to(device)
    src_predictor.eval()
    #
    tgt_generator_file = osp.join(args.root, args.model, args.name, '{}.tgt.generator.pth'.format(params.src_dataset))
    chk = torch.load(tgt_generator_file)
    state_dict = chk['model_state_dict']
    tgt_generator = Generator(dim=4, motiflen=20)
    tgt_generator.load_state_dict(state_dict)
    tgt_generator.to(device)
    tgt_generator.eval()
    #
    print("{}: {} Adaptation for {}".format(args.name, params.src_dataset, params.tgt_dataset))
    out_dir = osp.join(args.root, 'chr1/location')
    locating_adap(tgt_generator, src_predictor, t_loader, out_dir, args.name + '.' + params.src_dataset + '.adap.st')


if __name__ == "__main__":
    main()

