#!/usr/bin/bash

# Splitting Genome into non-overlapping bins of 600bp
bedtools makewindows -g ../Genome/hg38.chrom1.size -w 600 -s 600 > ./hg38.chrom1.size.bins
cat ./hg38.chrom1.size.bins | wc -l
bedtools intersect -a ./hg38.chrom1.size.bins -b ../Genome/hg38.blacklist.regions.bed -v > ./Human.chrom1.size.bins.filtered
rm -f ./hg38.chrom1.size.bins
cat ./Human.chrom1.size.bins.filtered | wc -l
# Mouse
bedtools makewindows -g ../Genome/mm10.chrom1.size -w 600 -s 600 > ./mm10.chrom1.size.bins
cat ./mm10.chrom1.size.bins | wc -l
bedtools intersect -a ./mm10.chrom1.size.bins -b ../Genome/mm10.blacklist.regions.bed -v > ./Mouse.chrom1.size.bins.filtered
rm -f ./mm10.chrom1.size.bins
cat ./Mouse.chrom1.size.bins.filtered | wc -l
