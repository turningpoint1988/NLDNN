#!/usr/bin/bash

ROOT="${HOME}/NLDNN"

for target in $(ls ${ROOT}/TF-specific/)
do
    echo "Working on ${target}."
    # visualization of contributions of SNPs
    
    python visualization.py -r ${ROOT} -n ${target}
  
done

