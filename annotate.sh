#!/bin/bash

ROOT="${HOME}/NLDNN"
#
for target in $(ls ./Human-Mouse/ )
do
    echo "Working on ${target}."

    # annotate
    bedtools intersect -a ${ROOT}/Genome/hg38.chrom.sizes.bins.filtered -b ${ROOT}/Human-Mouse/${target}/ChIPseq.Human.${target}.idr.bed -f 0.2 -c > ${ROOT}/Human-Mouse/${target}/ChIPseq.Human.${target}.temp
    bedtools intersect -a ${ROOT}/Human-Mouse/${target}/ChIPseq.Human.${target}.temp -b ${ROOT}/Human-Mouse/${target}/ChIPseq.Human.${target}.idr.bed -c > ${ROOT}/Human-Mouse/${target}/ChIPseq.Human.${target}.bins
    rm -f ${ROOT}/Human-Mouse/${target}/ChIPseq.Human.${target}.temp
    cat ${ROOT}/Human-Mouse/${target}/ChIPseq.Human.${target}.bins | head -n 10
    echo "Human annotation done!"
    # Mouse
    bedtools intersect -a ${ROOT}/Genome/mm10.chrom.sizes.bins.filtered -b ${ROOT}/Human-Mouse/${target}/ChIPseq.Mouse.${target}.idr.bed -f 0.2 -c > ${ROOT}/Human-Mouse/${target}/ChIPseq.Mouse.${target}.temp
    bedtools intersect -a ${ROOT}/Human-Mouse/${target}/ChIPseq.Mouse.${target}.temp -b ${ROOT}/Human-Mouse/${target}/ChIPseq.Mouse.${target}.idr.bed -c > ${ROOT}/Human-Mouse/${target}/ChIPseq.Mouse.${target}.bins
    rm -f ${ROOT}/Human-Mouse/${target}/ChIPseq.Mouse.${target}.temp
    cat ${ROOT}/Human-Mouse/${target}/ChIPseq.Mouse.${target}.bins | head -n 10
    echo "Mouse annotation done!"
#
#    # data pre-process
    if [ ! -d ${ROOT}/Human-Mouse/${target}/raw_data/hg38 ]; then
        mkdir -p ${ROOT}/Human-Mouse/${target}/raw_data/hg38
    fi

    if [ ! -d ${ROOT}/Human-Mouse/${target}/raw_data/mm10 ]; then
        mkdir -p ${ROOT}/Human-Mouse/${target}/raw_data/mm10
    fi
    python data_pre.py -r ${ROOT} -t ${target}

    # spliting data into training, validation, and test sets
    bash ${ROOT}/make_val_test_file.sh "${ROOT}" "$target" "hg38" || exit 1
    bash ${ROOT}/make_val_test_file.sh "${ROOT}" "$target" "mm10" || exit 1

    # encoding sequences into one-hot matrces, due to the large amount of data, it is stored in batches.
    python encode.py -r ${ROOT} -t ${target}

    # recording the number of stored batches,
    num_human_tr=$(ls ${ROOT}/Human-Mouse/${target}/data | grep "Human.${target}.tr.neg" | wc -l)
    num_human_te=$(ls ${ROOT}/Human-Mouse/${target}/data | grep "Human.${target}.te.neg" | wc -l)
    num_human_va=$(ls ${ROOT}/Human-Mouse/${target}/data | grep "Human.${target}.va.neg" | wc -l)

    num_mouse_tr=$(ls ${ROOT}/Human-Mouse/${target}/data | grep "Mouse.${target}.tr.neg" | wc -l)
    num_mouse_te=$(ls ${ROOT}/Human-Mouse/${target}/data | grep "Mouse.${target}.te.neg" | wc -l)
    num_mouse_va=$(ls ${ROOT}/Human-Mouse/${target}/data | grep "Mouse.${target}.va.neg" | wc -l)

    echo -e "${target}\t${num_human_tr}\t${num_human_te}\t${num_human_va}\t${num_mouse_tr}\t${num_mouse_te}\t${num_mouse_va}" >> ${ROOT}/count.txt
done
