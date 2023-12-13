#!/usr/bin/python
import os
import sys
import argparse
import random
import numpy as np
import os.path as osp
from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("..")
import params


def Readfimo(file):
    pvalues = []
    with open(file) as f:
        f.readline()
        for line in f:
            line_split = line.strip().split()
            if len(line_split) != 9:
                continue
            # chrom = line_split[1]
            # start = line_split[2]
            # end = line_split[3]
            # strand = line_split[4]
            pv = -np.log10(float(line_split[6]))
            pvalues.append(pv)
    return pvalues


def readpos(bed, sequence_dict):
    with open(bed) as f:
        lines = f.readlines()
    seqs = []
    positions = []
    for line in lines:
        line_split = line.strip().split()
        chrom = line_split[0]
        start = int(line_split[1])
        end = int(line_split[2])
        seq = str(sequence_dict[chrom].seq[start:end]).upper()
        seqs.append(seq)
        position = int(line_split[3])
        positions.append(position)
    return seqs, positions


def readneg(bed, sequence_dict, num):
    with open(bed) as f:
        lines = f.readlines()
    index = list(range(len(lines)))
    index_s = random.sample(index, num)
    seqs = []
    for i in index_s:
        line_split = lines[i].strip().split()
        chrom = line_split[0]
        start = int(line_split[1])
        end = int(line_split[2])
        seq = str(sequence_dict[chrom].seq[start:end]).upper()
        seqs.append(seq)
    return seqs


def count(loc_pos, temp_dir, remap):
    with open(osp.join(temp_dir, 'temp.bed'), 'w') as f:
        for i in loc_pos:
            f.write("chr1\t{}\t{}\n".format(i, i+1))
    os.system('bedtools intersect -a {} -b {} -u > {}'.format(osp.join(temp_dir, 'temp.bed'), remap,
                                                              osp.join(temp_dir, 'temp_overlap.bed')))

    with open(osp.join(temp_dir, 'temp_overlap.bed')) as f:
        lines = f.readlines()

    return len(lines)


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="The ratio of located peaks to total peaks.")

    parser.add_argument("-r", dest="root", type=str, default=None,
                        help="The path of the project.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of one of Cell.TF.")

    return parser.parse_args()


args = get_args()


def source_source(score_f):
    neg_bed = args.name + '.' + params.src_dataset + '.ss.neg.bed'
    pos_bed = args.name + '.' + params.src_dataset + '.ss.pos.bed'
    if params.src_dataset == 'Human':
        ref = osp.join(args.root, 'Genome/hg38.fa')
    else:
        ref = osp.join(args.root, 'Genome/mm10.fa')
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(ref), 'fasta'))
    seq_pos, loc_pos = readpos(osp.join(args.root, 'chr1/location', pos_bed), sequence_dict)
    # count
    name = args.name.split('.')[1]
    if params.src_dataset == 'Human':
        remap = osp.join(args.root, 'chr1/remap/remap2022_{}_chr1_hg38.bed'.format(name))
    else:
        remap = osp.join(args.root, 'chr1/remap/remap2022_{}_chr1_mm10.bed'.format(name))

    overlap = count(loc_pos, osp.join(args.root, 'chr1/location'), remap)
    #
    seq_neg = readneg(osp.join(args.root, 'chr1/location', neg_bed), sequence_dict, len(loc_pos))
    #
    pos_fa = args.name + '.' + params.src_dataset + '.ss.pos.fa'
    with open(osp.join(args.root, 'chr1/location', pos_fa), 'w') as f:
        for i in range(len(seq_pos)):
            f.write('>seq_pos_{}\n'.format(i + 1))
            f.write('{}\n'.format(seq_pos[i]))
    neg_fa = args.name + '.' + params.src_dataset + '.ss.neg.fa'
    with open(osp.join(args.root, 'chr1/location', neg_fa), 'w') as f:
        for i in range(len(seq_neg)):
            f.write('>seq_neg_{}\n'.format(i + 1))
            f.write('{}\n'.format(seq_neg[i]))
    #
    PWM = osp.join(args.root, 'PWMs/{}/{}.{}.meme'.format(name, params.src_dataset, name))
    if not osp.exists(osp.join(args.root, 'chr1/fimo/{}'.format(args.name))):
        os.mkdir(osp.join(args.root, 'chr1/fimo/{}'.format(args.name)))
    print("excuting the fimo program for pos data....")
    fimo_out = osp.join(args.root, 'chr1/fimo/{}/{}_ss_pos'.format(args.name, params.src_dataset))
    os.system('fimo --bfile --uniform-- --max-stored-scores 10000 --max-strand '
              '--thresh 1e-04 --oc {} {} {}'.format(fimo_out, PWM, osp.join(args.root, 'chr1/location', pos_fa)))
    fimo_file = osp.join(fimo_out, 'fimo.tsv')
    pos_pv = Readfimo(fimo_file)
    #
    print("excuting the fimo program for neg data....")
    fimo_out = osp.join(args.root, 'chr1/fimo/{}/{}_ss_neg'.format(args.name, params.src_dataset))
    os.system('fimo --bfile --uniform-- --max-stored-scores 10000 --max-strand '
              '--thresh 1e-04 --oc {} {} {}'.format(fimo_out, PWM, osp.join(args.root, 'chr1/location', neg_fa)))
    fimo_file = osp.join(fimo_out, 'fimo.tsv')
    neg_pv = Readfimo(fimo_file)
    total = len(loc_pos)
    print("The overlap ratio between predicted peaks and true peaks is {:.3f}/({}/{})".format(overlap / total, overlap,
                                                                                              total))
    print("The number of instances in positive and control are {} and {}\n".format(len(pos_pv), len(neg_pv)))
    with open(score_f, 'a') as f:
        f.write("{}\t{}\t{}\t{:.3f}\t{}\t{}\t{:.3f}\n".format(args.name, overlap, total, overlap/total, len(pos_pv),
                                                              len(neg_pv), np.log(len(pos_pv)/len(neg_pv))))


def source_target(score_f):
    pos_bed = args.name + '.' + params.src_dataset + '.st.pos.bed'
    pos_bed_adap = args.name + '.' + params.src_dataset + '.adap.st.pos.bed'
    if params.tgt_dataset == 'Human':
        ref = osp.join(args.root, 'Genome/hg38.fa')
    else:
        ref = osp.join(args.root, 'Genome/mm10.fa')
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(ref), 'fasta'))
    seq_pos, loc_pos = readpos(osp.join(args.root, 'chr1/location', pos_bed), sequence_dict)
    seq_pos_adap, loc_pos_adap = readpos(osp.join(args.root, 'chr1/location', pos_bed_adap), sequence_dict)
    #
    name = args.name.split('.')[1]
    if params.tgt_dataset == 'Human':
        remap = osp.join(args.root, 'chr1/remap/remap2022_{}_chr1_hg38.bed'.format(name))
    else:
        remap = osp.join(args.root, 'chr1/remap/remap2022_{}_chr1_mm10.bed'.format(name))
    overlap = count(loc_pos, osp.join(args.root, 'chr1/location'), remap)
    overlap_adap = count(loc_pos_adap, osp.join(args.root, 'chr1/location'), remap)
    #
    pos_fa = args.name + '.' + params.src_dataset + '.st.pos.fa'
    with open(osp.join(args.root, 'chr1/location', pos_fa), 'w') as f:
        for i in range(len(seq_pos)):
            f.write('>seq_pos_{}\n'.format(i + 1))
            f.write('{}\n'.format(seq_pos[i]))
    pos_fa_adap = args.name + '.' + params.src_dataset + '.adap.st.pos.fa'
    with open(osp.join(args.root, 'chr1/location', pos_fa_adap), 'w') as f:
        for i in range(len(seq_pos_adap)):
            f.write('>seq_pos_adap_{}\n'.format(i + 1))
            f.write('{}\n'.format(seq_pos_adap[i]))
    #
    PWM = osp.join(args.root, 'PWMs/{}/{}.{}.meme'.format(name, params.tgt_dataset, name))
    if not osp.exists(osp.join(args.root, 'chr1/fimo/{}'.format(args.name))):
        os.mkdir(osp.join(args.root, 'chr1/fimo/{}'.format(args.name)))
    print("excuting the fimo program for pos data....")
    fimo_out = osp.join(args.root, 'chr1/fimo/{}/{}_st_pos'.format(args.name, params.src_dataset))
    os.system('fimo --bfile --uniform-- --max-stored-scores 10000 --max-strand '
              '--thresh 1e-04 --oc {} {} {}'.format(fimo_out, PWM, osp.join(args.root, 'chr1/location', pos_fa)))
    fimo_file = osp.join(fimo_out, 'fimo.tsv')
    pos_pv = Readfimo(fimo_file)
    #
    print("excuting the fimo program for pos adap data....")
    fimo_out = osp.join(args.root, 'chr1/fimo/{}/{}_adap_st_pos'.format(args.name, params.src_dataset))
    os.system('fimo --bfile --uniform-- --max-stored-scores 10000 --max-strand '
              '--thresh 1e-04 --oc {} {} {}'.format(fimo_out, PWM, osp.join(args.root, 'chr1/location', pos_fa_adap)))
    fimo_file = osp.join(fimo_out, 'fimo.tsv')
    pos_adap_pv = Readfimo(fimo_file)
    #
    total = len(loc_pos)
    print("The overlap ratio of pos sequences is {:.3f}/({}/{})".format(overlap / total, overlap, total))
    print("The overlap ratio of pos adap sequences is {:.3f}/({}/{})".format(overlap_adap / total, overlap_adap, total))
    print("The number of instances in pos and pos adap are {} and {}\n".format(len(pos_pv), len(pos_adap_pv)))
    with open(score_f, 'a') as f:
        f.write("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(args.name, overlap / total, overlap_adap / total,
                                                              len(pos_pv) / total, len(pos_adap_pv) / total))


def main():
    # model_name = 'NLDNN'
    # print("{}: {} for {}".format(args.name, params.src_dataset, params.src_dataset))
    # score_f = osp.join(args.root, 'chr1/{}.{}.ss.score.txt'.format(model_name, params.src_dataset))
    # source_source(score_f)
    print("{}: Source vs. Adaptation".format(args.name))
    score_f = osp.join(args.root, 'chr1/{}.{}.st.adap.score.txt'.format('NLDNN', params.src_dataset))
    source_target(score_f)


if __name__ == "__main__":
    main()

