#!/usr/bin/bash

ROOT="${HOME}/NLDNN"
MODEL="models_NLDNN"
CELL=('Lymphoblast' 'liver' 'Erythroid')
TF=('BHLHE40' 'CTCF' 'E2F4' 'JUND' 'MYC' 'GATA1' 'MAFK' 'RXRA')

for cell in ${CELL[*]}
do
    for tf in ${TF[*]}
    do
        echo "Working on ${cell}.${tf}."
        if [ ! -d ../${MODEL}/"${cell}.${tf}" ]; then
            echo "${cell}.${tf} does not exist."
            continue
        fi
        # before performing SNP classification, we should do data preparation
        python encode.py  -r ${ROOT} -t ${cell} -n ${tf}
        # for all nucleotide-level models
        python predict.py -r ${ROOT} -t ${cell} -n ${tf} -w 51 -m ${MODEL}
        # only for NLDNN and NLDNN-AT
        python predict_at.py -r ${ROOT} -t ${cell} -n ${tf} -w 51 -m ${MODEL}
        
    done
  
done

