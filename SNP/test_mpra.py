#!/usr/bin/python

import os
import numpy as np
import os.path as osp
from Bio import SeqIO

import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import h5py
import argparse

# custom functions defined by user
import sys
sys.path.append("..")
from model import NLDNN
from datasets import TargetDataSet


def one_hot(seq):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


def readfile(infile, sequence_dict, seq_len):
    ref_data = []
    alt_data = []
    effect = []
    with open(infile) as f:
        f.readline()
        for line in f:
            line_split = line.strip().split()
            chrom = "chr" + line_split[0]
            position = int(line_split[1])
            ref = line_split[2]
            alt = line_split[3]
            value = float(line_split[-2])
            position -= 1
            nucleotide = str(sequence_dict[chrom].seq[position:position + 1])
            assert nucleotide == ref, "nucleotide is not consistent"
            start = position - seq_len // 2
            end = position + seq_len // 2 + 1
            seq_ref = str(sequence_dict[chrom].seq[start:end])
            seq_alt = seq_ref[:(seq_len // 2)] + alt + seq_ref[(seq_len // 2 + 1):]
            assert (seq_ref[seq_len // 2] == ref) and (seq_alt[seq_len // 2] == alt), "position is not consistent"
            ref_data.append(one_hot(seq_ref))
            alt_data.append(one_hot(seq_alt))
            effect.append(value)
    ref_data = np.array(ref_data, dtype=np.float32)
    ref_data = ref_data.transpose((0, 2, 1))
    alt_data = np.array(alt_data, dtype=np.float32)
    alt_data = alt_data.transpose((0, 2, 1))

    return ref_data, alt_data, effect


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build NLDNN for MPRA data.")

    parser.add_argument("-r", dest="root", type=str, default=None)
    parser.add_argument("-m", dest="model", type=str, default=None)

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    root = args.root
    model_name = args.model
    name = model_name.split('_')[-1]
    dim = 4
    window = 51
    seq_len = 601
    model_dict = {'HepG2': {'F9': 'chrX', 'LDLR': 'chr19', 'SORT1': 'chr1'},
                  'Panc1': {'TCF7L2': 'chr10', 'ZFAND3': 'chr6'},
                  'HEK293T': {'HNF4A': 'chr20', 'MSMB': 'chr10', 'TERT-HEK': 'chr5', 'MYCrs6983267': 'chr8'},
                  'SK-MEL-S': {'IRF4': 'chr6'},
                  'K562': {'PKLR-48h': 'chr1'},
                  'GM12878': {'GP1BA': 'chr22', 'HBB': 'chr11', 'HBG1': 'chr11'},
                  'NHEK': {'IRF6': 'chr1'}
                  }
    device = torch.device("cuda:0")
    target = model_dict[name]
    for key in target.keys():
        f_out = open(root + '/SNP/mpra.txt', 'a')
        # Load weights
        checkpoint_file = root + '/{}/{}/Human.model.best.pth'.format(model_name, target[key])
        chk = torch.load(checkpoint_file, map_location='cuda:0')
        state_dict = chk['model_state_dict']
        model = NLDNN(dim=dim, motiflen=20)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        # Load data
        sequence_dict = SeqIO.to_dict(
            SeqIO.parse(open(root + '/Genome/hg38.fa'), 'fasta'))
        infile = root + 'SNP/MPRA/GRCh38_{}.tsv'.format(key)
        ref_data, alt_data, effects = readfile(infile, sequence_dict, seq_len)
        #
        ref_loader = DataLoader(TargetDataSet(ref_data), batch_size=1, shuffle=False, num_workers=0)
        alt_loader = DataLoader(TargetDataSet(alt_data), batch_size=1, shuffle=False, num_workers=0)
        # score_log = []
        score_local = []
        for ref_batch, alt_batch in zip(ref_loader, alt_loader):
            ref = ref_batch["data"].float().to(device)
            alt = alt_batch["data"].float().to(device)
            with torch.no_grad():
                ref_p = model(ref)
                alt_p = model(alt)
            ref_p = ref_p.view(-1).data.cpu().numpy()
            alt_p = alt_p.view(-1).data.cpu().numpy()
            # score_log.append(np.log2(np.sum(alt_p) / np.sum(ref_p)))
            position = seq_len // 2
            score_local.append(np.sum(alt_p[(position - window // 2):(position + window // 2)]) -
                               np.sum(ref_p[(position - window // 2):(position + window // 2)]))

        # pr, _ = pearsonr(effects, score_log)
        # spr, _ = spearmanr(effects, score_log)
        # print("{}\t{}\tpearsonr: {:.3f}\tspearmanr: {:.3f}\n\n".format(name, key, pr, spr))
        # f_out.write("{}\t{}\tpearsonr: {:.3f}\tspearmanr: {:.3f}\n\n".format(name, key, pr, spr))

        pr, _ = pearsonr(effects, score_local)
        spr, _ = spearmanr(effects, score_local)
        print("{}\t{}\tpearson: {:.3f}\tspearman: {:.3f}\n\n".format(name, key, pr, spr))
        f_out.write("{}\t{}\tpearson: {:.3f}\tspearman: {:.3f}\n\n".format(name, key, pr, spr))
        f_out.close()


if __name__ == "__main__":
    main()

