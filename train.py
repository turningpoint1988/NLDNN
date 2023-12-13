#!/usr/bin/python

import os
import sys
import argparse
import random
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# custom functions defined by user
from model import NLDNN
from datasets import SourceDataSet
from loss import MSELoss
import h5py
import params


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, max_epoch,
                 data_path, name, batch_size, scheduler, checkpoint, dim):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.data_path = data_path
        self.name = name
        self.dim = dim
        with h5py.File(osp.join(self.data_path, '{}.{}.tr.pos.hdf5'.format(params.src_dataset, self.name)), 'r') as f:
            self.data_tr_pos = np.array(f['data'], dtype=np.float32)
            self.label_tr_pos = np.array(f['signal'], dtype=np.float32)
        with h5py.File(osp.join(self.data_path, '{}.{}.va.pos.hdf5'.format(params.src_dataset, self.name)), 'r') as f:
            data_va_pos = np.array(f['data'], dtype=np.float32)
            label_va_pos = np.array(f['signal'], dtype=np.float32)
        self.batch_size = batch_size
        self.val_loader = DataLoader(SourceDataSet(data_va_pos, label_va_pos), batch_size=self.batch_size * 10,
                                     shuffle=False, num_workers=0)
        self.max_epoch = max_epoch
        self.scheduler = scheduler
        self.checkpoint = checkpoint
        self.rmse = 10000

    def train(self):
        """training the model"""
        self.model.to(self.device)
        if params.src_dataset == 'Human':
            tr_neg_num = params.neg_count[self.name]['human.tr.neg']
            index = list(range(tr_neg_num))
        else:
            tr_neg_num = params.neg_count[self.name]['mouse.tr.neg']
            index = list(range(tr_neg_num))

        if self.max_epoch <= len(index):
            index = random.sample(index, self.max_epoch)
        else:
            random.shuffle(index)
            index = index + random.sample(index * (self.max_epoch // len(index)),
                                          self.max_epoch - len(index))
        for epoch in range(self.max_epoch):
            index_neg = index[epoch]
            print("We are sampling the {}-th negative part".format(index_neg))
            # build training data generator
            with h5py.File(osp.join(self.data_path, '{}.{}.tr.neg{}.hdf5'.format(
                    params.src_dataset, self.name, index_neg)), 'r') as f:
                data_tr_neg = np.array(f['data'], dtype=np.float32)
                label_tr_neg = np.array(f['signal'], dtype=np.float32)
            data_tr = np.concatenate((self.data_tr_pos, data_tr_neg), axis=0)
            label_tr = np.concatenate((self.label_tr_pos, label_tr_neg), axis=0)
            train_data = SourceDataSet(data_tr, label_tr)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
            # set training mode during the training process
            self.model.train()
            for i_batch, sample_batch in enumerate(train_loader):
                x_data = sample_batch["data"].float().to(self.device)
                label = sample_batch["label"].float().to(self.device)
                pred = self.model(x_data)
                loss = self.criterion(pred, label)
                if np.isnan(loss.item()):
                    raise ValueError('loss is nan while training')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (i_batch + 1) % len(train_loader) == 0:
                    print("epoch/i_batch: {}/{}---loss: {:.4f}".format(epoch + 1, i_batch + 1, loss.item()))
            # validation and save the model with higher accuracy
            if (epoch + 1) % params.eval_step_pre == 0:
                rmse = self.validation()
                if self.rmse > rmse:
                    print("Store the weights of the model in the current run.\n")
                    self.rmse = rmse
                    checkpoint_file = osp.join(self.checkpoint, '{}.model.best.pth'.format(params.src_dataset))
                    torch.save({'model_state_dict': self.model.state_dict()}, checkpoint_file)
            self.scheduler.step()

        # return self.rmse

    def validation(self):
        """validate the performance of the trained model."""
        self.model.eval()
        loss_all = []
        for i_batch, sample_batch in enumerate(self.val_loader):
            X_data = sample_batch["data"].float().to(self.device)
            label = sample_batch["label"].float().to(self.device)
            with torch.no_grad():
                pred = self.model(X_data)
                loss = self.criterion(pred, label)
            loss_all.append(loss.item())
        # neg
        if params.src_dataset == 'Human':
            va_neg_num = params.neg_count[self.name]['human.va.neg']
            index = list(range(va_neg_num))
        else:
            va_neg_num = params.neg_count[self.name]['mouse.va.neg']
            index = list(range(va_neg_num))
        for i in index:
            with h5py.File(osp.join(self.data_path, '{}.{}.va.neg{}.hdf5'.format(
                    params.src_dataset, self.name, i)), 'r') as f:
                data_va_neg = np.array(f['data'], dtype=np.float32)
                label_va_neg = np.array(f['signal'], dtype=np.float32)
            val_loader_neg = DataLoader(SourceDataSet(data_va_neg, label_va_neg), batch_size=self.batch_size,
                                        shuffle=False, num_workers=0)

            for i_batch, sample_batch in enumerate(val_loader_neg):
                x_data = sample_batch["data"].float().to(self.device)
                label = sample_batch["label"].float().to(self.device)
                with torch.no_grad():
                    pred = self.model(x_data)
                    loss = self.criterion(pred, label)
                loss_all.append(loss.item())
        rmse = np.mean(loss_all)
        return rmse


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
    dim = args.dimension
    random.seed(params.manual_seed)
    # implement
    lr = 0.001
    betas = (0.9, 0.999)
    weight_decay = 0.00001
    criterion = MSELoss()
    model = NLDNN(dim=dim, motiflen=20)
    # loading the best model
    checkpoint_file = osp.join(args.checkpoint, '{}.model.pth'.format(params.src_dataset))
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    executor = Trainer(model=model,
                       optimizer=optimizer,
                       criterion=criterion,
                       device=params.device,
                       max_epoch=params.num_epochs_pre,
                       data_path=args.data_dir,
                       name=args.name,
                       batch_size=params.batch_size,
                       scheduler=scheduler,
                       checkpoint=args.checkpoint,
                       dim=dim)
    executor.train()


if __name__ == "__main__":
    main()

