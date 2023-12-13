#!/usr/bin/python

import os
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import h5py
import argparse

# custom functions defined by user
import sys
sys.path.append("..")
from model import NLDNN
from datasets import TargetDataSet


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build NLDNN for TF-specific SNPs classification")

    parser.add_argument("-r", dest="root", type=str, default=None)
    parser.add_argument("-t", dest="target", type=str, default=None)
    parser.add_argument("-n", dest="name", type=str, default=None)
    parser.add_argument("-m", dest="model", type=str, default=None)
    parser.add_argument("-w", dest="window", type=int, default=51)

    return parser.parse_args()


def main():
    args = get_args()
    root = args.root
    target = args.target
    name = args.name
    model_name = args.model
    window = args.window
    dim = 4
    seq_len = 601
    device = torch.device("cuda:0")
    f_out = open(root + '/SNP/snp.score.txt', 'a')
    # Load weights
    checkpoint_file = root + '/{}/{}.{}/Human.model.best.pth'.format(model_name, target, name)
    chk = torch.load(checkpoint_file, map_location='cuda:0')
    state_dict = chk['model_state_dict']
    model = NLDNN(dim=dim, motiflen=20)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # Load data
    with h5py.File(root + '/SNP/TF-specific/{}.{}/data/snp_data.hdf5'.format(target, name), 'r') as f:
        pos_ref = np.array(f['pos_ref'], dtype=np.float32)
        # pos_ref = pos_ref[:,:,:-1]
        pos_alt = np.array(f['pos_alt'], dtype=np.float32)
        # pos_alt = pos_alt[:,:,:-1]
        neg_ref = np.array(f['neg_ref'], dtype=np.float32)
        # neg_ref = neg_ref[:,:,:-1]
        neg_alt = np.array(f['neg_alt'], dtype=np.float32)
        # neg_alt = neg_alt[:,:,:-1]
    #
    pos_ref_loader = DataLoader(TargetDataSet(pos_ref), batch_size=1, shuffle=False, num_workers=0)
    pos_alt_loader = DataLoader(TargetDataSet(pos_alt), batch_size=1, shuffle=False, num_workers=0)
    score_all_pos = []
    for ref_batch, alt_batch in zip(pos_ref_loader, pos_alt_loader):
        ref_data = ref_batch["data"].float().to(device)
        alt_data = alt_batch["data"].float().to(device)
        with torch.no_grad():
            ref_p = model(ref_data)
            alt_p = model(alt_data)
        ref_p = ref_p.view(-1).data.cpu().numpy()
        alt_p = alt_p.view(-1).data.cpu().numpy()
        position = seq_len // 2
        score_all_pos.append(np.sum(np.abs(ref_p[(position-window//2):(position+window//2)] -
                                           alt_p[(position-window//2):(position+window//2)])))

    neg_ref_loader = DataLoader(TargetDataSet(neg_ref), batch_size=1, shuffle=False, num_workers=0)
    neg_alt_loader = DataLoader(TargetDataSet(neg_alt), batch_size=1, shuffle=False, num_workers=0)
    score_all_neg = []
    for ref_batch, alt_batch in zip(neg_ref_loader, neg_alt_loader):
        ref_data = ref_batch["data"].float().to(device)
        alt_data = alt_batch["data"].float().to(device)
        with torch.no_grad():
            ref_p = model(ref_data)
            alt_p = model(alt_data)
        ref_p = ref_p.view(-1).data.cpu().numpy()
        alt_p = alt_p.view(-1).data.cpu().numpy()
        position = seq_len // 2
        score_all_neg.append(np.sum(np.abs(ref_p[(position-window//2):(position+window//2)] -
                                           alt_p[(position-window//2):(position+window//2)])))

    pred_all = score_all_pos + score_all_neg
    label = [1] * len(score_all_pos) + [0] * len(score_all_neg)
    print("The baseline is {:.3f}\n".format(len(score_all_pos)/len(score_all_neg)))
    print("Running by {}\n".format(model_name))
    prauc = average_precision_score(label, pred_all)
    auc = roc_auc_score(label, pred_all)
    print("{}.{}\tauc: {:.3f}\tprauc: {:.3f}\t{}\n".format(target, name, auc, prauc, model_name.split('_')[-1]))
    f_out.write("{}.{}\tauc: {:.3f}\tprauc: {:.3f}\t{}\n".format(target, name, auc, prauc, model_name.split('_')[-1]))
    f_out.close()


if __name__ == "__main__":
    main()

