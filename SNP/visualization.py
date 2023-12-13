from __future__ import division, print_function
import h5py
import os
import argparse
import numpy as np
import os.path as osp
import viz_sequence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def lineplot(seq_ref, seq_alt, out_f):
    x = list(range(len(seq_ref))) * 2
    y = np.concatenate((seq_ref, seq_alt))
    type = ['Ref'] * len(seq_ref) + ['Alt'] * len(seq_alt)
    df = pd.DataFrame({'x': x, 'y': y, 'type': type})
    sns.set_theme(style="white")
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 2))
    # sns.despine(fig)
    sns.lineplot(
        data=df,
        x="x", y="y", hue='type', dashes=False, ax=ax1, linewidth=1)
    ax1.set_ylabel("Coverage")
    ax1.set_xlabel("")
    ax1.set_xmargin(0)
    ax1.legend(title=None, frameon=True)
    #
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(out_f, format='png', bbox_inches='tight', dpi=300)


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualization of contributions of SNPs.")

    parser.add_argument("-r", dest="root", type=str, default=None)
    parser.add_argument("-n", dest="name", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    root = args.root
    name = args.name
    # load data
    with h5py.File(osp.join(root, 'SNP/TF-specific/{}/data/pos.contrib.hdf5'.format(name)), 'r') as f:
        contrib_ref = np.array(f['contrib_ref'], dtype=np.float32)
        contrib_alt = np.array(f['contrib_alt'], dtype=np.float32)
        seq_ref = np.array(f['pred_ref'], dtype=np.float32)
        seq_alt = np.array(f['pred_alt'], dtype=np.float32)

    # visualization
    # background = np.sum(np.sum(seqs, axis=1), axis=0) / np.sum(np.sum(np.sum(seqs, axis=1), axis=0))
    number = 2
    for i in range(number):
        seq_ref_one = seq_ref[i]
        seq_alt_one = seq_alt[i]
        lineplot(seq_ref_one, seq_alt_one, root + '/SNP/TF-specific/{}/pred_{}.png'.format(name, i))
        contrib_ref_one = contrib_ref[i]
        contrib_alt_one = contrib_alt[i]
        viz_sequence.plot_weights(contrib_ref_one[250:350, :], subticks_frequency=20, figsize=(20, 2),
                                  out_f=root + '/SNP/TF-specific/{}/contrib_ref_{}_part.png'.format(name, i)
                                  )
        viz_sequence.plot_weights(contrib_alt_one[250:350, :], subticks_frequency=20, figsize=(20, 2),
                                  out_f=root + '/SNP/TF-specific/{}/contrib_alt_{}_part.png'.format(name, i)
                                  )
        viz_sequence.plot_weights(contrib_ref_one, subticks_frequency=50, figsize=(20, 2),
                                  out_f=root + '/SNP/TF-specific/{}/contrib_ref_{}.png'.format(name, i)
                                  )
        viz_sequence.plot_weights(contrib_alt_one, subticks_frequency=50, figsize=(20, 2),
                                  out_f=root + '/SNP/TF-specific/{}/contrib_alt_{}.png'.format(name, i)
                                  )
