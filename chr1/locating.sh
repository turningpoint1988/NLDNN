#!/usr/bin/bash

# before performing location, we should do data preparation

bash make_windows.sh

ROOT="${HOME}/NLDNN"

array=('Lymphoblast.CTCF' 'Lymphoblast.E2F4' 'Lymphoblast.JUND' 'Lymphoblast.MYC' 'Lymphoblast.BHLHE40' 'Erythroid.BHLHE40' 'Erythroid.CTCF' 'Erythroid.E2F4' 'Erythroid.GATA1' 'Erythroid.JUND' 'Erythroid.MAFK' 'Erythroid.MYC' 'liver.CTCF' 'liver.RXRA' 'ESC.CTCF' 'ESC.MAFK' 'ESC.NANOG')


for experiment in ${array[*]}
do
    echo "working on ${experiment}."
    python locating.py -r ${ROOT} \
                       -n $experiment \
                       -m "models_NLDNN" #| tee -a ./predictbychrom.log.txt
    
    python ratio.py -r ${ROOT} \
                    -n ${experiment} #| tee -a ./fimo.log.txt 
done
