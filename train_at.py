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
from model import Generator, Predictor, Discriminator
from adapt import train_tgt_at, train_tgt_tf, test_tgt
import params


def init_model(net, restore=None):
    if restore is not None and osp.exists(restore):
        print("Loading pre-trained weights.")
        checkpoint = torch.load(restore, map_location='cuda:0')
        state_dict = checkpoint["model_state_dict"]
        net.load_state_dict(state_dict, strict=False)
    net.to(params.device)
    return net


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Adversarial training of NLDNN.")

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
    src_generator = init_model(net=Generator(dim=dim), restore=checkpoint_file)
    src_predictor = init_model(net=Predictor(size=params.seq_size), restore=checkpoint_file)
    f = open(osp.join(args.checkpoint, 'score.txt'), 'a')
    # train target generator by adversarial training
    print("=== Training generator for target domain ===")
    best_prauc = 0
    best_generator = None
    for ratio in params.ratio:
        for clip_value in params.clip_value:
            hyperparams = {'ratio': ratio, 'clip_value': clip_value}
            print("We are using ratio:{} and clip_value:{}".format(ratio, clip_value))
            prauc_all = []
            pr_all = []
            for trial in range(params.trial):
                # init weights of target generator with those of source generator
                tgt_generator = init_model(net=Generator(dim=dim))
                tgt_generator.load_state_dict(src_generator.state_dict())
                discriminator = init_model(Discriminator())
                f.write(">>> {} adaption for {} <<<\n".format(params.src_dataset, params.tgt_dataset))
                prauc, pr, tgt_generator = train_tgt_at(src_generator, tgt_generator, src_predictor, discriminator,
                                                        args.data_dir, hyperparams, args.name)
                prauc_all.append(prauc)
                pr_all.append(pr)
            f.write("(ratio: {} clip_value: {})\tprauc: {:.3f}\tpearson: {:.3f}\n\n".format(
                ratio, clip_value, np.mean(prauc_all), np.mean(pr_all)))
            f.flush()
            if best_prauc < np.mean(prauc_all):
                best_prauc = np.mean(prauc_all)
                best_generator = deepcopy(tgt_generator)
    f.close()
    # save models
    checkpoint_file = osp.join(args.checkpoint, '{}.tgt.generator.pth'.format(params.src_dataset))
    torch.save({'model_state_dict': best_generator.state_dict()}, checkpoint_file)
    checkpoint_file = osp.join(args.checkpoint, '{}.src.predictor.pth'.format(params.src_dataset))
    torch.save({'model_state_dict': src_predictor.state_dict()}, checkpoint_file)
    

if __name__ == "__main__":
    main()

