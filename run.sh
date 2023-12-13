#!/usr/bin/bash

ROOT="${HOME}/NLDNN"

for target in $(ls ${ROOT}/Human-Mouse/)
do
    echo "Working on ${target}."
    if [ ! -d ${ROOT}/models_NLDNN/${target} ]; then
        mkdir -p ${ROOT}/models_NLDNN/${target}
    fi
    # training
    if [ ! -f ${ROOT}/models_NLDNN/${target}/Human.model.pth ]; then
        echo ">> Starting to warm up the model. <<"
        python warm_up.py -d ${ROOT}/Human-Mouse/${target}/data \
                        -n ${target} \
                        -dim 4 \
                        -c ${ROOT}/models_NLDNN/${target}
        echo ">> Warming up is finished. <<"
    fi

    echo ">> Starting to train the model. <<"
    python train.py -d ${ROOT}/Human-Mouse/${target}/data \
                    -n ${target} \
                    -dim 4 \
                    -c ${ROOT}/models_NLDNN/${target}
    echo ">> Training is finished. <<"
    # testing
    echo ">> Starting to test the model. <<"
    python test.py -d ${ROOT}/Human-Mouse/${target}/data \
                   -n ${target} \
                   -dim 4 \
                   -c ${ROOT}/models_NLDNN/${target}
    echo ">> Testing is finished.<<"

done

