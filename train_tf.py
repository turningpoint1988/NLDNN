#!/usr/bin/python

import os
import sys
import argparse
import random
import numpy as np
import os.path as osp
from copy import deepcopy
import torch

# custom functions defined by user
from model import Transfer, Generator, Predictor
from adapt import train_tgt_tf, test_tgt
import params


def init_model(net, restore=None):
    if restore is not None and osp.exists(restore):
        print("Loading pre-trained weights.")
        checkpoint = torch.load(restore)
        state_dict = checkpoint["model_state_dict"]
        net.load_state_dict(state_dict, strict=False)
    net.to(params.device)
    return net


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Transfer Learning for cross-species prediction.")

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
    # loading the best model
    checkpoint_file = osp.join(args.checkpoint, '{}.model.best.pth'.format(params.src_dataset))
    if not osp.exists(checkpoint_file):
        print("The pretrained model is not existed.")
        sys.exit(0)
    tgt_generator = init_model(net=Generator(dim=dim), restore=checkpoint_file)
    src_predictor = init_model(net=Predictor(size=params.seq_size), restore=checkpoint_file)
    f = open(osp.join(args.checkpoint, 'score.txt'), 'a')
    # train target generator by transfer learning
    print("=== Training generator by transfer learning ===")
    best_prauc = 0
    best_generator = None
    for ratio in params.ratio:
        print("We are using ratio:{}".format(ratio))
        hyperparams = {'ratio': ratio}
        # init weights of target generator with those of source generator
        transfer = init_model(Transfer())
        f.write(">>> {} transfer for {} <<<\n".format(params.src_dataset, params.tgt_dataset))
        prauc, pr, tgt_generator = train_tgt_tf(tgt_generator, src_predictor, transfer,
                                                args.data_dir, hyperparams, args.name)
        f.write("(ratio: {})\tprauc: {:.3f}\tpearson: {:.3f}\n\n".format(
                ratio, prauc, pr))
        f.flush()
        if best_prauc < prauc:
            best_prauc = prauc
            best_generator = deepcopy(tgt_generator)
    f.close()
    # save models
    checkpoint_file = osp.join(args.checkpoint, '{}.tgt.generator.pth'.format(params.src_dataset))
    torch.save({'model_state_dict': best_generator.state_dict()}, checkpoint_file)
    

if __name__ == "__main__":
    main()

