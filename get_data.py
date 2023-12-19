#!/usr/bin/env python

import os, argparse, sys
import os.path as osp


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-i", dest="inputfile", type=str, default='')
    parser.add_argument("-o", dest="outdir", type=str, default='')

    return parser.parse_args()


def download(inputfile, outdir):
    with open(inputfile) as f:
        lines = f.readlines()
    
    if not osp.exists(outdir):
            os.mkdir(outdir)
    for line in lines:
        line_split = line.strip().split()
        if len(line_split) == 0:
            continue
        name = line_split[0]
        accession = line_split[1]
        suffix = line_split[2]
        #
        name_s = name.split(".")
        name_o = name_s[1] + '.' + name_s[2]

        if not osp.exists(outdir + '/' + name_o):
            os.mkdir(outdir + '/' + name_o)

        if suffix == 'bed.gz':
            print("downloading peak files for {}...".format(name))
            url = 'https://www.encodeproject.org/files/{}/@@download/{}.{}'.format(accession, accession, suffix)
            outfile = outdir + '/ChIPseq.{}.bed.gz'.format(name)
            os.system('curl -o {} -J -L {}'.format(outfile, url))
            os.system('gunzip {}'.format(outfile))
            os.system('mv -f {} {}'.format(outdir + '/ChIPseq.{}.bed'.format(name), outdir + '/' + name_o))
        elif suffix == 'bigWig':
            print("downloading bigWig files for {}...".format(name))
            url = 'https://www.encodeproject.org/files/{}/@@download/{}.{}'.format(accession, accession, suffix)
            outfile = outdir + '/ChIPseq.{}.bigWig'.format(name)
            os.system('curl -o {} -J -L {}'.format(outfile, url))
            os.system('mv -f {} {}'.format(outdir + '/ChIPseq.{}.bigWig'.format(name), outdir + '/' + name_o))
        elif suffix == 'bam':
            print("downloading bigWig files for {}...".format(name))
            url = 'https://www.encodeproject.org/files/{}/@@download/{}.{}'.format(accession, accession, suffix)
            outfile = outdir + '/ChIPseq.{}.bam'.format(name)
            os.system('curl -o {} -J -L {}'.format(outfile, url))
            os.system('mv -f {} {}'.format(outdir + '/ChIPseq.{}.bam'.format(name), outdir + '/' + name_o))
        else:
            print("error occurs, no valid suffix.")
            sys.exit(0)


args = get_args()
download(args.inputfile, args.outdir)




