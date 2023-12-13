# coding:utf-8
import os.path as osp
import os
import random
import numpy as np
import h5py
import argparse


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
def outputHDF5(pos_ref, pos_alt, neg_ref, neg_alt, filename):
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(filename, 'w') as f:
        f.create_dataset('pos_ref', data=pos_ref, **comp_kwargs)
        f.create_dataset('pos_alt', data=pos_alt, **comp_kwargs)
        f.create_dataset('neg_ref', data=neg_ref, **comp_kwargs)
        f.create_dataset('neg_alt', data=neg_alt, **comp_kwargs)


def encode(snp, snp_ld, out_f):
    pos_ref = []
    pos_alt = []
    with open(snp) as f:
        lines = f.readlines()
    for line in lines:
        line_split = line.strip().split()
        ref = line_split[-2]
        ref_one_hot = one_hot(ref)
        pos_ref.append(ref_one_hot)
        alt = line_split[-1]
        alt_one_hot = one_hot(alt)
        pos_alt.append(alt_one_hot)
    pos_ref = np.array(pos_ref, dtype=np.float32)
    pos_ref = pos_ref.transpose((0, 2, 1))
    pos_alt = np.array(pos_alt, dtype=np.float32)
    pos_alt = pos_alt.transpose((0, 2, 1))
    neg_ref = []
    neg_alt = []
    with open(snp_ld) as f:
        lines = f.readlines()
    for line in lines:
        line_split = line.strip().split()
        ref = line_split[-2]
        ref_one_hot = one_hot(ref)
        neg_ref.append(ref_one_hot)
        alt = line_split[-1]
        alt_one_hot = one_hot(alt)
        neg_alt.append(alt_one_hot)
    neg_ref = np.array(neg_ref, dtype=np.float32)
    neg_ref = neg_ref.transpose((0, 2, 1))
    neg_alt = np.array(neg_alt, dtype=np.float32)
    neg_alt = neg_alt.transpose((0, 2, 1))
    outputHDF5(pos_ref, pos_alt, neg_ref, neg_alt, out_f)


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Encode TF-specific SNPs.")

    parser.add_argument("-r", dest="root", type=str, default=None)
    parser.add_argument("-t", dest="target", type=str, default=None)
    parser.add_argument("-n", dest="name", type=str, default=None)

    return parser.parse_args()


def main():
    args = get_args()
    root = args.root
    target = args.target
    name = args.name
    data_dir = root + '/SNP/TF-specific/{}.{}'.format(target, name)
    out_dir = osp.join(data_dir, 'data')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    snp = data_dir + '/positive.bed'.format(target, name)
    snp_ld = data_dir + '/negative.bed'.format(target, name)
    out_f = out_dir + '/snp_data.hdf5'
    encode(snp, snp_ld, out_f)


if __name__ == '__main__':  main()
