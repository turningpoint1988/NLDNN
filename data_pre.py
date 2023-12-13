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


def dataselect(file, sequence_dict):
    """selecting all binding and non-binding sequences from the whole genome"""
    dict_pos = {}
    dict_neg = {}
    with open(file) as f:
        for line in f:
            line_split = line.strip().split()
            chrom = line_split[0]
            start = int(line_split[1])
            end = int(line_split[2])
            flag1 = int(line_split[3])
            flag2 = int(line_split[4])
            seq = list(str(sequence_dict[chrom].seq[start:end]).upper())
            if seq.count('N') > SEQ_LEN*0.01: continue
            gc = (seq.count('G') + seq.count('C')) / SEQ_LEN
            gc = round(gc, 3)
            if flag1 > 0:
                if chrom not in dict_pos.keys():
                    dict_pos[chrom] = [(start, end, gc)]
                else:
                    dict_pos[chrom].append((start, end, gc))
            else:
                if flag2 > 0: continue
                if chrom not in dict_neg.keys():
                    dict_neg[chrom] = [(start, end, gc)]
                else:
                    dict_neg[chrom].append((start, end, gc))

    return dict_pos, dict_neg


def datastore(dict_pos, dict_neg, out_f):
    """filtering out those not matching the GC distribution of pos sequences,
    and then store all pos and neg sequences."""
    bin_size = 0.01
    f = open(out_f, 'w')
    for key, value in dict_pos.items():
        print("we are working on {}.".format(key))
        gc_pos_num = [0] * int(1 / bin_size)
        for elem in value:
            index = int(elem[-1]/bin_size)
            gc_pos_num[index] += 1
            f.write("{}\t{}\t{}\t{}\n".format(key, elem[0], elem[1], 1))
        value_neg = dict_neg[key]
        gc_neg_num = [0] * len(gc_pos_num)
        for elem in value_neg:
            index = int(elem[-1] / bin_size)
            if gc_pos_num[index] > 0:
                gc_neg_num[index] += 1
                f.write("{}\t{}\t{}\t{}\n".format(key, elem[0], elem[1], 0))
    f.close()


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CNN for predicting CSSBS")

    parser.add_argument("-r", dest="root", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-t", dest="target", type=str, default=None,
                        help="A directory containing the training data.")

    return parser.parse_args()


def main():
    random.seed(666)
    args = get_args()
    root = args.root
    target = args.target
    genome = root + '/Genome'
    data_dir = root + '/Human-Mouse/{}'.format(target)
    # Human
    human_out = osp.join(root, 'Human-Mouse/{}/raw_data/hg38/all.all'.format(target))
    seq_bed = data_dir + '/ChIPseq.Human.{}.bins'.format(target)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome + '/hg38.fa'), 'fasta'))
    dict_pos, dict_neg = dataselect(seq_bed, sequence_dict)
    datastore(dict_pos, dict_neg, human_out)
    # Mouse
    mouse_out = osp.join(root, 'Human-Mouse/{}/raw_data/mm10/all.all'.format(target))
    seq_bed = data_dir + '/ChIPseq.Mouse.{}.bins'.format(target)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(genome + '/mm10.fa'), 'fasta'))
    dict_pos, dict_neg = dataselect(seq_bed, sequence_dict)
    datastore(dict_pos, dict_neg, mouse_out)


if __name__ == '__main__':  main()
