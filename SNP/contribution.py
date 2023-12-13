from __future__ import division, print_function
import numpy as np
import h5py
import os
import torch
from captum.attr import DeepLift, IntegratedGradients
import os.path as osp
import argparse
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
import params
from datasets import SourceDataSet, TargetDataSet
from model import NLDNN
from model_contrib import NLDNN_Contrib


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCN for motif location")

    parser.add_argument("-r", dest="root", type=str, default=None)
    parser.add_argument("-n", dest="name", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    root = args.root
    name = args.name
    src_dataset = 'Human'
    dim = 4
    position = 300
    window = 51
    device = torch.device("cuda:0")
    #
    checkpoint_file = osp.join(root, 'models_NLDNN/{}/{}.model.best.pth'.format(name, src_dataset))
    chk = torch.load(checkpoint_file, map_location="cuda:0")
    state_dict = chk['model_state_dict']
    model = NLDNN(dim=dim, motiflen=20)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    #
    with h5py.File(root + '/SNP/TF-specific/{}/data/snp_data.hdf5'.format(name), 'r') as f:
        seqs_ref = np.array(f['pos_ref'], dtype=np.float32)  # Nx4x601
        seqs_alt = np.array(f['pos_alt'], dtype=np.float32)  # Nx4x601

    print("The number of positive samples is {}".format(len(seqs_ref)))
    data_ref = TargetDataSet(seqs_ref)
    ref_loader = DataLoader(data_ref, batch_size=len(seqs_ref), shuffle=False, num_workers=0)
    data_alt = TargetDataSet(seqs_alt)
    alt_loader = DataLoader(data_alt, batch_size=len(seqs_alt), shuffle=False, num_workers=0)
    scores = []
    for ref_batch, alt_batch in zip(ref_loader, alt_loader):
        ref = ref_batch["data"].float().to(device)
        alt = alt_batch["data"].float().to(device)
        with torch.no_grad():
            pred_ref = model(ref)
            pred_alt = model(alt)
        print(pred_ref.size())
        pred_ref = pred_ref.view(pred_ref.size()[0], -1).data.cpu().numpy()
        pred_alt = pred_alt.view(pred_alt.size()[0], -1).data.cpu().numpy()
        #
        scores = np.sum(np.abs(pred_ref[:, (position - window // 2):(position + window // 2)] -
                               pred_alt[:, (position - window // 2):(position + window // 2)]), axis=1)

    index = np.argsort(scores)
    index = index[::-1]
    pred_ref = pred_ref[index]
    pred_alt = pred_alt[index]
    seqs_ref = seqs_ref[index]
    seqs_alt = seqs_alt[index]
    #
    device = torch.device("cpu")
    model = NLDNN_Contrib(dim=dim, motiflen=20)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    dl = DeepLift(model)
    contribution_ref = []
    contribution_alt = []
    for ref, alt in zip(seqs_ref, seqs_alt):
        data_one = torch.from_numpy(ref)
        data_one = data_one.view(1, data_one.size()[0], data_one.size()[1])
        data_one.requires_grad = True
        ref_one = torch.zeros_like(data_one)
        #
        model.zero_grad()
        contribution = dl.attribute(data_one, ref_one, target=0, return_convergence_delta=False)
        contribution = torch.mean(contribution, dim=0)
        contribution = contribution.data.numpy()
        contribution_ref.append(contribution)
        # alt
        data_one = torch.from_numpy(alt)
        data_one = data_one.view(1, data_one.size()[0], data_one.size()[1])
        data_one.requires_grad = True
        ref_one = torch.zeros_like(data_one)
        model.zero_grad()
        contribution = dl.attribute(data_one, ref_one, target=0, return_convergence_delta=False)
        contribution = torch.mean(contribution, dim=0)
        contribution = contribution.data.numpy()
        contribution_alt.append(contribution)

    contribution_ref = np.array(contribution_ref, dtype=np.float32)
    contribution_ref = contribution_ref.transpose((0, 2, 1))
    contribution_alt = np.array(contribution_alt, dtype=np.float32)
    contribution_alt = contribution_alt.transpose((0, 2, 1))
    #
    seqs_ref = seqs_ref.transpose((0, 2, 1))
    seqs_alt = seqs_alt.transpose((0, 2, 1))
    # store the results
    f = h5py.File(root + '/SNP/TF-specific/{}/data/pos.contrib.hdf5'.format(name), 'w')
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    f.create_dataset('seq_ref', data=seqs_ref, **comp_kwargs)
    f.create_dataset('seq_alt', data=seqs_alt, **comp_kwargs)
    f.create_dataset('pred_ref', data=pred_ref, **comp_kwargs)
    f.create_dataset('pred_alt', data=pred_alt, **comp_kwargs)
    f.create_dataset('contrib_ref', data=contribution_ref, **comp_kwargs)
    f.create_dataset('contrib_alt', data=contribution_alt, **comp_kwargs)
    f.close()
