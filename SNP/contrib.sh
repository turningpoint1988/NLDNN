#!/usr/bin/bash

ROOT="${HOME}/NLDNN"

for target in $(ls ${ROOT}/SNP/TF-specific/)
do
    echo "Working on ${target}."
    # computing the contributions of SNPs
    
    python contribution.py -r ${ROOT} -n ${target} 
  
done

