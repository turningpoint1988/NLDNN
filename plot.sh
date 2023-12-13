#!/usr/bin/bash


ROOT="${HOME}/NLDNN"

for target in $(ls ./models_NLDNN/)
do
    echo "Working on ${target}."
    # plot contributions and ISSM values for DNA regions
    python plot_contributions.py -r ${ROOT} \
                                 -n ${target} \
                                 -m "models_NLDNN"
                               
    # plot predictions for DNA regions
    python plot_predictions.py -r ${ROOT} \
                               -n ${target} \
                               -m "models_NLDNN"
done

