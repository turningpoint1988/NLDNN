#!/usr/bin/python

import argparse
import random
import sys

import numpy as np
import os.path as osp

from model import NLDNN
from datasets import SourceDataSet
from loss import MSELoss
import h5py
import params

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, max_epoch,
                 data_path, name, batch_size, dim):
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
        self.rmse = 0

    def train(self):
        """training the model"""
        self.model.to(self.device)
        for epoch in range(self.max_epoch):
            index_neg = epoch
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
            # validation and save the model with higher accuracy
            self.validation()

        return self.rmse, self.model.state_dict()

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
        self.rmse = np.mean(loss_all)


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
    random.seed(params.manual_seed)
    device = params.device
    name = args.name
    dim = args.dimension
    # implement
    rmse_lowest = 10000
    criterion = MSELoss()
    for i in range(8):
        print("The program is working on the {}-th round".format(i+1))
        model = NLDNN(dim=dim, motiflen=20)
        optimizer = optim.Adam(model.parameters(), lr=0.001,
                               betas=(0.9, 0.999),
                               weight_decay=0.00001)
        executor = Trainer(model=model,
                           optimizer=optimizer,
                           criterion=criterion,
                           device=device,
                           max_epoch=3,
                           data_path=args.data_dir,
                           name=name,
                           batch_size=params.batch_size,
                           dim=dim)

        rmse, state_dict = executor.train()
        if rmse_lowest > rmse:
            print("Store the weights of the model in the current run.\n")
            rmse_lowest = rmse
            checkpoint_file = osp.join(args.checkpoint, '{}.model.pth'.format(params.src_dataset))
            torch.save({
                'model_state_dict': state_dict,
                'parameter_state_dice': optimizer.param_groups[0]
            }, checkpoint_file)


if __name__ == "__main__":
    main()

