#!/usr/bin/bash

DATABASE="HUMAN"

ROOT="${HOME}/NLDNN"

for target in $(ls ${ROOT}/Human-Mouse/)
do
    echo "working on $target."
    if [ ! -d ${ROOT}/motifs/$target/$DATABASE ]; then
        mkdir -p ${ROOT}/motifs/$target/$DATABASE
    fi
    
    python motif.py -d ${ROOT}/Human-Mouse/$target/data \
                    -n $target \
                    -t 0.5 \
                    -c ${ROOT}/models_NLDNN/$target \
                    -o ${ROOT}/motifs/$target/$DATABASE
                           
    # tomtom
    tomtom -oc ${ROOT}/motifs/$target/$DATABASE -evalue -thresh 5 -dist pearson -min-overlap 5 -no-ssc -verbosity 1 ${ROOT}/motifs/$target/$DATABASE/motif.meme ${ROOT}/motif_databases/${DATABASE}/HOCOMOCOv11_core_${DATABASE}_mono_meme_format.meme #-png
    
done

