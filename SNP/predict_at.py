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
from model import Generator, Predictor, NLDNN
from datasets import TargetDataSet


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build NLDNN-AT for TF-specific SNPs classification")

    parser.add_argument("-r", dest="root", type=str, default=None)
    parser.add_argument("-t", dest="target", type=str, default=None)
    parser.add_argument("-n", dest="name", type=str, default=None)
    parser.add_argument("-m", dest="model", type=str, default=None)
    parser.add_argument("-w", dest="window", type=int, default=51)

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    root = args.root
    target = args.target
    name = args.name
    window = args.window
    model_name = args.model
    dim = 4
    seq_len = 601
    device = torch.device("cuda:0")
    f_out = open(root + '/SNP/snp.adap.score.txt', 'a')
    # Load data
    with h5py.File(root + '/SNP/TF-specific/{}.{}/data/snp_data.hdf5'.format(target, name), 'r') as f:
        pos_ref = np.array(f['pos_ref'], dtype=np.float32)
        pos_alt = np.array(f['pos_alt'], dtype=np.float32)
        neg_ref = np.array(f['neg_ref'], dtype=np.float32)
        neg_alt = np.array(f['neg_alt'], dtype=np.float32)

    # Load weights
    checkpoint_file = root + '/{}/{}.{}/Mouse.model.best.pth'.format(model_name, target, name)
    chk = torch.load(checkpoint_file, map_location='cuda:0')
    state_dict = chk['model_state_dict']
    model = NLDNN(dim=dim, motiflen=20)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    #
    # Load weights
    generator_file = root + '/{}/{}.{}/Mouse.tgt.generator.pth'.format(model_name, target, name)
    chk = torch.load(generator_file)
    state_dict = chk['model_state_dict']
    generator = Generator(dim=dim, motiflen=20)
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    #
    predictor_file = root + '/{}/{}.{}/Mouse.src.predictor.pth'.format(model_name, target, name)
    chk = torch.load(predictor_file)
    state_dict = chk['model_state_dict']
    predictor = Predictor(size=seq_len)
    predictor.load_state_dict(state_dict)
    predictor.to(device)
    predictor.eval()
    #
    pos_ref_loader = DataLoader(TargetDataSet(pos_ref), batch_size=1, shuffle=False, num_workers=0)
    pos_alt_loader = DataLoader(TargetDataSet(pos_alt), batch_size=1, shuffle=False, num_workers=0)
    score_all_pos = []
    score_all_pos_adap = []
    for ref_batch, alt_batch in zip(pos_ref_loader, pos_alt_loader):
        ref_data = ref_batch["data"].float().to(device)
        alt_data = alt_batch["data"].float().to(device)
        with torch.no_grad():
            ref_p_adap = predictor(generator(ref_data))
            alt_p_adap = predictor(generator(alt_data))
            #
            ref_p = model(ref_data)
            alt_p = model(alt_data)
        ref_p_adap = ref_p_adap.view(-1).data.cpu().numpy()
        alt_p_adap = alt_p_adap.view(-1).data.cpu().numpy()
        ref_p = ref_p.view(-1).data.cpu().numpy()
        alt_p = alt_p.view(-1).data.cpu().numpy()
        position = seq_len // 2
        score_all_pos_adap.append(np.sum(np.abs(ref_p_adap[(position - window // 2):(position + window // 2)] -
                                                alt_p_adap[(position - window // 2):(position + window // 2)])))
        score_all_pos.append(np.sum(np.abs(ref_p[(position - window // 2):(position + window // 2)] -
                                           alt_p[(position - window // 2):(position + window // 2)])))

    neg_ref_loader = DataLoader(TargetDataSet(neg_ref), batch_size=1, shuffle=False, num_workers=0)
    neg_alt_loader = DataLoader(TargetDataSet(neg_alt), batch_size=1, shuffle=False, num_workers=0)
    score_all_neg = []
    score_all_neg_adap = []
    for ref_batch, alt_batch in zip(neg_ref_loader, neg_alt_loader):
        ref_data = ref_batch["data"].float().to(device)
        alt_data = alt_batch["data"].float().to(device)
        with torch.no_grad():
            ref_p_adap = predictor(generator(ref_data))
            alt_p_adap = predictor(generator(alt_data))
            #
            ref_p = model(ref_data)
            alt_p = model(alt_data)
        ref_p_adap = ref_p_adap.view(-1).data.cpu().numpy()
        alt_p_adap = alt_p_adap.view(-1).data.cpu().numpy()
        ref_p = ref_p.view(-1).data.cpu().numpy()
        alt_p = alt_p.view(-1).data.cpu().numpy()
        position = seq_len // 2
        score_all_neg_adap.append(np.sum(np.abs(ref_p_adap[(position - window // 2):(position + window // 2)] -
                                                alt_p_adap[(position - window // 2):(position + window // 2)])))
        score_all_neg.append(np.sum(np.abs(ref_p[(position-window//2):(position+window//2)] -
                                           alt_p[(position-window//2):(position+window//2)])))

    pred_all = score_all_pos + score_all_neg
    pred_all_adap = score_all_pos_adap + score_all_neg_adap
    label = [1] * len(score_all_pos) + [0] * len(score_all_neg)
    prauc = average_precision_score(label, pred_all)
    auc = roc_auc_score(label, pred_all)
    print("{}.{}\tprauc: {:.3f}\tauc: {:.3f}\n".format(target, name, prauc, auc))
    f_out.write("{}.{}\tauc: {:.3f}\tprauc: {:.3f}\t{}\n".format(target, name, auc, prauc, "NLDNN"))
    ##
    prauc = average_precision_score(label, pred_all_adap)
    auc = roc_auc_score(label, pred_all_adap)
    print("{}.{}\tprauc: {:.3f}\tauc: {:.3f}\n".format(target, name, prauc, auc))
    f_out.write("{}.{}\tauc: {:.3f}\tprauc: {:.3f}\t{}\n".format(target, name, auc, prauc, "NLDNN-AT"))
    f_out.close()


if __name__ == "__main__":
    main()

