"""Adversarial adaptation to train target generator."""

import os.path as osp
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr

from datasets import SourceDataSet, TargetDataSet
from loss import MSELoss, BCELoss
import h5py
import params
import random
from copy import deepcopy


def train_tgt_at(src_generator, tgt_generator, src_predictor, discriminator, data_dir, hyperparams, name):
    ####################
    # 1. load data #
    ####################
    with h5py.File(osp.join(data_dir, '{}.{}.tr.pos.hdf5'.format(params.src_dataset, name)), 'r') as f:
        s_data_tr_pos = np.array(f['data'], dtype=np.float32)

    # target dataset
    with h5py.File(osp.join(data_dir, '{}.{}.tr.pos.hdf5'.format(params.tgt_dataset, name)), 'r') as f:
        t_data_tr_pos = np.array(f['data'], dtype=np.float32)

    if params.tgt_dataset == 'Human':
        tr_neg_num = params.neg_count[name]['human.tr.neg']
        index = list(range(tr_neg_num))
    else:
        tr_neg_num = params.neg_count[name]['mouse.tr.neg']
        index = list(range(tr_neg_num))

    index_neg = random.sample(index, 1)[0]
    print("randomly sampling the {}-th neg from {}".format(index_neg, params.tgt_dataset))
    with h5py.File(osp.join(data_dir, '{}.{}.tr.neg{}.hdf5'.format(params.tgt_dataset, name, index_neg)), 'r') as f:
        t_data_tr_neg = np.array(f['data'], dtype=np.float32)

    s_data_tr = s_data_tr_pos
    # # adjust the ratio of pos
    t_data_tr_pos_num = t_data_tr_pos.shape[0]
    sample_pos_num = int(t_data_tr_pos_num * hyperparams['ratio'])
    smaple_neg_num = t_data_tr_pos_num - sample_pos_num
    index = list(range(t_data_tr_pos_num))
    index_pos = random.sample(index, sample_pos_num)
    t_data_tr_neg_num = t_data_tr_neg.shape[0]
    index = list(range(t_data_tr_neg_num))
    index_neg = random.sample(index, smaple_neg_num)
    t_data_tr_pos = t_data_tr_pos[index_pos]
    t_data_tr_neg = t_data_tr_neg[index_neg]
    t_data_tr = np.concatenate((t_data_tr_pos, t_data_tr_neg))
    # construct dataloader
    src_dataset = TargetDataSet(s_data_tr)
    src_data_loader = DataLoader(src_dataset, batch_size=params.batch_size,
                                 shuffle=True, num_workers=0)
    tgt_dataset = TargetDataSet(t_data_tr)
    tgt_data_loader = DataLoader(tgt_dataset, batch_size=params.batch_size,
                                 shuffle=True, num_workers=0)
    # iterable
    src_data_loader_iter = iter(src_data_loader)
    tgt_data_loader_iter = iter(tgt_data_loader)
    ####################
    # 1. setup network #
    ####################
    # setup criterion and optimizer
    sigmoid = torch.nn.Sigmoid()
    criterion = BCELoss()
    optimizer_tgt = optim.Adam(tgt_generator.parameters(),
                               lr=params.t_learning_rate,
                               betas=(params.beta1, params.beta2),
                               weight_decay=params.weight_decay)
    optimizer_discriminator = optim.Adam(discriminator.parameters(),
                                         lr=params.d_learning_rate,
                                         betas=(0.9, 0.999),
                                         weight_decay=params.weight_decay)

    ####################
    # 2. train network #
    ####################
    device = params.device
    source_label = 1
    target_label = 0
    prauc_best = 0
    pr_best = 0
    tgt_generator_best = None
    for epoch in range(params.num_epochs):
        ###########################
        # 2.1 train discriminator for multiple times
        ###########################
        # set state for discriminator
        discriminator.train()
        tgt_generator.eval()
        src_generator.eval()
        for d_step in range(params.d_train_step):
            # get source data
            try:
                src_batch = next(src_data_loader_iter)
                x_data_s = src_batch["data"].float().to(device)
                b = x_data_s.size()[0]
                label_s = torch.full((b, 1), source_label).float().to(device)
            except StopIteration:
                src_data_loader = DataLoader(src_dataset, batch_size=params.batch_size,
                                             shuffle=True, num_workers=0)
                src_data_loader_iter = iter(src_data_loader)
                continue
            # get target data
            try:
                tgt_batch = next(tgt_data_loader_iter)
                x_data_t = tgt_batch["data"].float().to(device)
                b = x_data_t.size()[0]
                label_t = torch.full((b, 1), target_label).float().to(device)
            except StopIteration:
                tgt_data_loader = DataLoader(tgt_dataset, batch_size=params.batch_size,
                                             shuffle=True, num_workers=0)
                tgt_data_loader_iter = iter(tgt_data_loader)
                continue
            # zero gradients for optimizer
            optimizer_discriminator.zero_grad()

            # extract and concat features
            feat_src = src_generator(x_data_s)
            feat_tgt = tgt_generator(x_data_t)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            feat_concat = feat_concat.detach()

            # predict on discriminator
            pred_concat = discriminator(feat_concat)
            pred_concat = sigmoid(pred_concat)

            # prepare real and fake label
            label_concat = torch.cat((label_s, label_t), 0)

            # compute loss for discriminator
            loss_discriminator = criterion(pred_concat, label_concat)
            loss_discriminator.backward()

            # optimize discriminator
            optimizer_discriminator.step()

            # truncate
            for p in discriminator.parameters():
                p.data.clamp_(-hyperparams['clip_value'], hyperparams['clip_value'])

            if (d_step+1) % 200 == 0:
              print("d_step: {}\td_loss: {:.5f}".format(d_step + 1, loss_discriminator.item()))
        ############################
        # 2.2 train target generator #
        ############################
        # set state for tgt_generator
        tgt_generator.train()
        for t_step, tgt_batch in enumerate(tgt_data_loader):
            # get target data
            x_data_t = tgt_batch["data"].float().to(device)

            # zero gradients for optimizer
            optimizer_discriminator.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_generator(x_data_t)

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)
            pred_tgt = sigmoid(pred_tgt)

            # prepare fake labels
            label_tgt = torch.full_like(pred_tgt, source_label)

            # compute loss for target generator
            loss_tgt = criterion(pred_tgt, label_tgt) * params.weight
            loss_tgt.backward()

            # optimize target generator
            optimizer_tgt.step()
            #
            if (t_step+1) % 100 == 0:
              print("t_step: {}\tg_loss: {:.5f}".format(t_step + 1, loss_tgt.item() / params.weight))
        prauc, pr = test(tgt_generator, src_predictor, data_dir, params.tgt_dataset, name)
        print("epoch-{}: {:.3f}\t{:.3f}\n".format(epoch, prauc, pr))
        if epoch < 1:
            prauc_best = prauc
            pr_best = pr
            tgt_generator_best = deepcopy(tgt_generator)
        else:
            if prauc > prauc_best:
                prauc_best = prauc
                pr_best = pr
                tgt_generator_best = deepcopy(tgt_generator)
            else:
                print("iteration stops at epoch-{}".format(epoch))
                break

    return prauc_best, pr_best, tgt_generator_best


def train_tgt_tf(tgt_generator, src_predictor, transfer, data_dir, hyperparams, name):
    """Train generator for target domain."""
    ####################
    # 1. load data #
    ####################
    with h5py.File(osp.join(data_dir, '{}.{}.tr.pos.hdf5'.format(params.src_dataset, name)), 'r') as f:
        s_data_tr_pos = np.array(f['data'], dtype=np.float32)

    s_data_tr = s_data_tr_pos
    s_label_tr = np.zeros((len(s_data_tr), 1))
    # target dataset
    with h5py.File(osp.join(data_dir, '{}.{}.tr.pos.hdf5'.format(params.tgt_dataset, name)), 'r') as f:
        t_data_tr_pos = np.array(f['data'], dtype=np.float32)

    if params.tgt_dataset == 'Human':
        tr_neg_num = params.neg_count[name]['human.tr.neg']
        index = list(range(tr_neg_num))
    else:
        tr_neg_num = params.neg_count[name]['mouse.tr.neg']
        index = list(range(tr_neg_num))

    index_neg = random.sample(index, 1)[0]
    print("randomly sampling the {}-th neg from {}".format(index_neg, params.tgt_dataset))
    with h5py.File(osp.join(data_dir, '{}.{}.tr.neg{}.hdf5'.format(params.tgt_dataset, name, index_neg)), 'r') as f:
        t_data_tr_neg = np.array(f['data'], dtype=np.float32)

    # # adjust the ratio of pos
    t_data_tr_pos_num = t_data_tr_pos.shape[0]
    sample_pos_num = int(t_data_tr_pos_num * hyperparams['ratio'])
    smaple_neg_num = t_data_tr_pos_num - sample_pos_num
    index = list(range(t_data_tr_pos_num))
    index_pos = random.sample(index, sample_pos_num)
    t_data_tr_neg_num = t_data_tr_neg.shape[0]
    index = list(range(t_data_tr_neg_num))
    index_neg = random.sample(index, smaple_neg_num)
    t_data_tr_pos = t_data_tr_pos[index_pos]
    t_data_tr_neg = t_data_tr_neg[index_neg]
    t_data_tr = np.concatenate((t_data_tr_pos, t_data_tr_neg))
    t_label_tr = np.zeros((len(t_data_tr), 1))
    # construct dataloader
    data_tr = np.concatenate((s_data_tr, t_data_tr))
    label_tr = np.concatenate((s_label_tr, t_label_tr))
    dataset = SourceDataSet(data_tr, label_tr)
    data_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=0)

    ####################
    # 1. setup network #
    ####################
    # setup criterion and optimizer
    criterion = BCELoss()
    optimizer_generator = optim.Adam(tgt_generator.parameters(),
                                     lr=1e-5)
    optimizer_transfer = optim.Adam(transfer.parameters(),
                                    lr=1e-3,
                                    weight_decay=0.00001)

    ####################
    # 2. train network #
    ####################
    device = params.device
    prauc_best = 0
    pr_best = 0
    tgt_generator_best = None
    for epoch in range(10):
        ###########################
        # 2.1 train discriminator for multiple times#
        ###########################
        # set state for discriminator
        tgt_generator.train()
        transfer.train()
        for i_batch, sample_batch in enumerate(data_loader):
            x_data = sample_batch["data"].float().to(device)
            label = sample_batch["label"].float().to(device)
            pred = transfer(tgt_generator(x_data))
            loss = criterion(pred.view(-1), label.view(-1))
            if np.isnan(loss.item()):
                raise ValueError('loss is nan while training')
            optimizer_generator.zero_grad()
            optimizer_transfer.zero_grad()
            loss.backward()
            optimizer_generator.step()
            optimizer_transfer.step()

        prauc, pr = test(tgt_generator, src_predictor, data_dir, params.tgt_dataset, name)
        print("epoch-{}: {:.3f}\t{:.3f}\n".format(epoch, prauc, pr))
        if epoch < 1:
            prauc_best = prauc
            pr_best = pr
            tgt_generator_best = deepcopy(tgt_generator)
        else:
            if prauc > prauc_best:
                prauc_best = prauc
                pr_best = pr
                tgt_generator_best = deepcopy(tgt_generator)
            else:
                print("iteration stops at epoch-{}".format(epoch))
                break

    return prauc_best, pr_best, tgt_generator_best


def test_tgt(generator, predictor, data_dir, target_name, name):
    # set eval state for Dropout and BN layers
    generator.eval()
    predictor.eval()

    # load data
    with h5py.File(osp.join(data_dir, '{}.{}.te.pos.hdf5'.format(target_name, name)), 'r') as f:
        data_te_pos = np.array(f['data'], dtype=np.float32)
        label_te_pos = np.array(f['signal'], dtype=np.float32)
    #
    data_te_pos_loader = DataLoader(SourceDataSet(data_te_pos, label_te_pos), batch_size=params.batch_size,
                                    shuffle=False, num_workers=0)
    max_pos = []
    max_pos_t = []
    for step, sample_batch in enumerate(data_te_pos_loader):
        x_data = sample_batch["data"].float().to(params.device)
        label = sample_batch["label"].float()
        label = label.view(label.size()[0], -1).data.cpu().numpy()
        with torch.no_grad():
            pred = predictor(generator(x_data))
        pred = pred.view(pred.size()[0], -1).data.cpu().numpy()
        if step == 0:
            max_pos = np.max(pred, axis=1)
            max_pos_t = np.max(label, axis=1)
        else:
            max_pos = np.concatenate((max_pos, np.max(pred, axis=1)))
            max_pos_t = np.concatenate((max_pos_t, np.max(label, axis=1)))

    pr = pearsonr(max_pos_t, max_pos)[0]
    print("{}\tpearson: {:.3f}\n".format(target_name, pr))

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
                pred = predictor(generator(x_data))
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

    return prauc, pr

