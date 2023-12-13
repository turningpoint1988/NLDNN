 #!/usr/bin/python

import os
import sys
import re
import argparse
import math
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader

# custom functions defined by user
sys.path.append("..")
from model import NLDNN
from datasets import SourceDataSet


def one_hot(seq):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build NLDNN for causal SNPs.")

    parser.add_argument("-r", dest="root", type=str, default=None)
    parser.add_argument("-m", dest="model", type=str, default=None)

    return parser.parse_args()


def main():
    device = torch.device("cuda:0")
    root = args.root
    model_name = args.model
    # myeloma: chr7; CLL: chr15; autoimmune: chr6
    dict = {'autoimmune': 'chr6', 'CLL': 'chr15', 'myeloma': 'chr7'}
    for target in dict.keys():
        # Load weights
        checkpoint_file = osp.join(root, '{}/{}/Human.model.best.pth'.format(model_name, dict[target]))
        chk = torch.load(checkpoint_file, map_location='cuda:0')
        state_dict = chk['model_state_dict']
        model = NLDNN(dim=4, motiflen=20)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        #
        snp_file = root + '/SNP/Causal/{}_snp_601bp.txt'.format(target)
        seq_len = 601
        window = 51
        with open(snp_file) as f:
            f.readline()
            lines = f.readlines()
        seqs_ref = []
        seqs_alt = []
        seqs_id = []
        for line in lines:
            line_split = line.strip().split()
            rsid = eval(line_split[0])
            seq = eval(line_split[4])
            seq_split = re.split('\[|\]', seq)
            allel = seq_split[1].split('/')
            ref = allel[0]
            seq_ref = seq_split[0] + ref + seq_split[2]
            alts = allel[1:]
            for alt in alts:
                seq_alt = seq_split[0] + alt + seq_split[2]
                seqs_ref.append(one_hot(seq_ref))
                seqs_alt.append(one_hot(seq_alt))
                seqs_id.append(rsid)

        seqs_ref = np.array(seqs_ref, dtype=np.float32)
        seqs_ref = seqs_ref.transpose((0, 2, 1))
        seqs_alt = np.array(seqs_alt, dtype=np.float32)
        seqs_alt = seqs_alt.transpose((0, 2, 1))

        f = open(osp.join(root, 'SNP/{}_{}.txt'.format(target, model_name.split('_')[1])), 'w')
        snp_data = SourceDataSet(seqs_ref, seqs_alt)
        test_loader = DataLoader(snp_data, batch_size=1, shuffle=False, num_workers=0)
        score_all = []
        score_single = []
        for i_batch, sample_batch in enumerate(test_loader):
            ref_data = sample_batch["data"].float().to(device)
            alt_data = sample_batch["label"].float().to(device)
            with torch.no_grad():
                pred_ref = model(ref_data)
                pred_alt = model(alt_data)
            ref_p = pred_ref.view(-1).data.cpu().numpy()
            alt_p = pred_alt.view(-1).data.cpu().numpy()
            position = seq_len // 2
            score_all.append(np.sum(np.abs(ref_p[(position - window // 2):(position + window // 2)] -
                                           alt_p[(position - window // 2):(position + window // 2)])))
            score_single.append(np.abs(alt_p[position] - ref_p[position]))
        # select top-k outputs for displaying
        index = np.argsort(np.array(score_all))
        index = index[::-1]
        for i in index:
            seq_id = seqs_id[i]
            score1 = score_all[i]
            score2 = score_single[i]
            f.write("{}\t{:.3f}\t{:.3f}\n".format(seq_id, score1, score2))
        f.close()


if __name__ == "__main__":
    main()

