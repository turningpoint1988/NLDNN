#!/bin/bash

ROOT="${HOME}/NLDNN"
# Splitting Genome into overlapping bins of 200bp by a stride 50
bedtools makewindows -g $ROOT/Genome/hg38.chrom.sizes -w 600 -s 100 > $ROOT/Genome/hg38.chrom.sizes.bins
cat $ROOT/Genome/hg38.chrom.sizes.bins | wc -l
bedtools intersect -a $ROOT/Genome/hg38.chrom.sizes.bins -b $ROOT/Genome/hg38.blacklist.regions.bed -v > $ROOT/Genome/hg38.chrom.sizes.bins.filtered
rm -f $ROOT/Genome/hg38.chrom.sizes.bins
cat $ROOT/Genome/hg38.chrom.sizes.bins.filtered | wc -l
# Mouse
bedtools makewindows -g $ROOT/Genome/mm10.chrom.sizes -w 600 -s 100 > $ROOT/Genome/mm10.chrom.sizes.bins
cat $ROOT/Genome/mm10.chrom.sizes.bins | wc -l
bedtools intersect -a $ROOT/Genome/mm10.chrom.sizes.bins -b $ROOT/Genome/mm10.blacklist.regions.bed -v > $ROOT/Genome/mm10.chrom.sizes.bins.filtered
rm -f $ROOT/Genome/mm10.chrom.sizes.bins
cat $ROOT/Genome/mm10.chrom.sizes.bins.filtered | wc -l
