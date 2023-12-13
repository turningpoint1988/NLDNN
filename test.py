#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader

# custom functions defined by user
from model import NLDNN
from loss import MSELoss
from datasets import SourceDataSet, TargetDataSet
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
import h5py
import params


def test(model, data_dir, target_name, name):
    # set eval state for Dropout and BN layers
    model.eval()
    # load data
    with h5py.File(osp.join(data_dir, '{}.{}.te.pos.hdf5'.format(target_name, name)), 'r') as f:
        data_te_pos = np.array(f['data'], dtype=np.float32)
        label_te_pos = np.array(f['signal'], dtype=np.float32)
    #
    data_te_pos_loader = DataLoader(SourceDataSet(data_te_pos, label_te_pos), batch_size=params.batch_size,
                                    shuffle=False, num_workers=0)
    max_pos = []
    max_pos_t = []
    p_all = []
    t_all = []
    for step, sample_batch in enumerate(data_te_pos_loader):
        x_data = sample_batch["data"].float().to(params.device)
        label = sample_batch["label"].float()
        label = label.view(label.size()[0], -1).data.cpu().numpy()
        with torch.no_grad():
            pred = model(x_data)
        pred = pred.view(pred.size()[0], -1).data.cpu().numpy()
        if step == 0:
            max_pos = np.max(pred, axis=1)
            max_pos_t = np.max(label, axis=1)
            p_all = pred
            t_all = label

        else:
            max_pos = np.concatenate((max_pos, np.max(pred, axis=1)))
            max_pos_t = np.concatenate((max_pos_t, np.max(label, axis=1)))
            p_all = np.concatenate((p_all, pred))
            t_all = np.concatenate((t_all, label))
    pr_full = 0
    for t_one, p_one in zip(t_all, p_all):
        pr_full += pearsonr(t_one, p_one)[0]
    pr_full /= len(t_all)
    pr = pearsonr(max_pos_t, max_pos)[0]
    print("{}\tfull pearson: {:.3f}\tpearson: {:.3f}\n".format(target_name, pr_full, pr))
    # neg
    max_neg_all = []
    if target_name == 'Human':
        te_neg_num = params.neg_count[name]['human.te.neg']
        index = list(range(te_neg_num))
    else:
        te_neg_num = params.neg_count[name]['mouse.te.neg']
        index = list(range(te_neg_num))
    for i in index:
        with h5py.File(osp.join(data_dir, '{}.{}.te.neg{}.hdf5'.format(target_name, name, i)), 'r') as f:
            data_te_neg = np.array(f['data'], dtype=np.float32)
        #
        data_te_neg_loader = DataLoader(TargetDataSet(data_te_neg), batch_size=1000, shuffle=False,
                                        num_workers=0)
        max_neg = []
        for step, sample_batch in enumerate(data_te_neg_loader):
            x_data = sample_batch["data"].float().to(params.device)
            with torch.no_grad():
                pred = model(x_data)
            pred = pred.view(pred.size()[0], -1).data.cpu().numpy()
            if step == 0:
                max_neg = np.max(pred, axis=1)
            else:
                max_neg = np.concatenate((max_neg, np.max(pred, axis=1)))

        if len(max_neg_all) == 0:
            max_neg_all = max_neg
        else:
            max_neg_all = np.concatenate((max_neg_all, max_neg))
    #
    pos_num = max_pos.shape[0]
    neg_num = max_neg_all.shape[0]
    print("pos: {}\t neg: {}".format(pos_num, neg_num))
    label = np.concatenate((np.ones(pos_num), np.zeros(neg_num)))
    prediciton = np.concatenate((max_pos, max_neg_all))
    prauc = average_precision_score(label, prediciton)
    print("{}\tprauc: {:.3f}\n".format(target_name, prauc))

    return prauc, pr_full, pr


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="A nucleotide-level model for predicting TF binding.")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="The directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of one of Cell.TF.")
    parser.add_argument("-dim", dest="dimension", type=int, default=4,
                        help="The channel number of input.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    name = args.name
    dim = args.dimension
    device = params.device
    f = open(osp.join(args.checkpoint, 'score.txt'), 'a')
    # Load weights
    checkpoint_file = osp.join(args.checkpoint, '{}.model.best.pth'.format(params.src_dataset))
    chk = torch.load(checkpoint_file, map_location='cuda:0')
    # return
    state_dict = chk['model_state_dict']
    model = NLDNN(dim=dim, motiflen=20)
    model.load_state_dict(state_dict)
    model.to(device)

    print(">>> source only for source <<<")
    prauc, pr_full, pr = test(model, args.data_dir, params.src_dataset, name)
    f.write(">>> {} only for {} <<<\n".format(params.src_dataset, params.src_dataset))
    f.write("{}\tprauc: {:.3f}\tpr_full: {:.3f}\tpr: {:.3f}\n\n".format(
        params.src_dataset, prauc, pr_full, pr))
    print(">>> source only for target <<<")
    prauc, pr_full, pr = test(model, args.data_dir, params.tgt_dataset, name)
    f.write(">>> {} only for {} <<<\n".format(params.src_dataset, params.tgt_dataset))
    f.write("{}\tprauc: {:.3f}\tpr_full: {:.3f}\tpr: {:.3f}\n\n".format(
        params.tgt_dataset, prauc, pr_full, pr))
    f.close()


if __name__ == "__main__":
    main()

