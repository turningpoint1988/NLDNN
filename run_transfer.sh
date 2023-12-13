#!/usr/bin/bash


ROOT="${HOME}/NLDNN"

for target in $(ls ${ROOT}/models_NLDNN/)
do
    echo "Working on ${target}."
    if [ ! -d ${ROOT}/models_NLDNN/${target} ]; then
        echo "${target} does not existed"
        exit
    fi
    
    echo ">> Starting to adversially train the model. <<"
    python train_tf.py -d ${ROOT}/Human-Mouse/${target}/data \
                       -n ${target} \
                       -dim 4 \
                       -c ${ROOT}/models_NLDNN/${target}
    echo ">> Training is finished. <<"
  
done


