# NLDNN-AT

**Cross-species prediction of transcription factor binding by adversarial training of a novel nucleotide-level deep neural network.** <br/>
The flowchart of NLDNN-AT is displayed as follows:

<p align="center"> 
<img src=https://github.com/turningpoint1988/NLDNN/blob/main/Pictures/flowchart.jpg>
</p>

<h4 align="center"> 
Fig.1 The flowchart of our proposed method (NLDNN-AT).
</h4>

## Prerequisites and Dependencies

- Pytorch 1.1 [[Install]](https://pytorch.org/)
- CUDA 9.0
- Python 3.6
- Python packages: biopython, scikit-learn, pyBigWig, h5py, scipy, pandas, matplotlib, seaborn

## Other Tools

- [MEME Suite](https://meme-suite.org/meme/doc/download.html): It assembles several methods used by this paper, including TOMTOM and FIMO.
- [Bedtools](https://bedtools.readthedocs.io/en/latest/content/installation.html): It is a powerful toolset for genome arithmetic.
- [TF–Modisco](https://github.com/jmschrei/tfmodisco-lite): It is a biological motif discovery algorithm that differentiates itself by using attribution scores from a machine learning model, in addition to the sequence itself, to guide motif discovery. 
- [Captum](https://github.com/pytorch/captum): It is a model interpretability and understanding library for PyTorch

## Competing Methods

- Sequence-level classification models: [LSGKM](https://github.com/Dongwon-Lee/lsgkm), [DeepSea](http://deepsea.princeton.edu/), [DanQ](https://github.com/uci-cbcl/DanQ), [DanQV](https://github.com/seqcode/cross-species-domain-adaptation).
- Sequence-level regression models: [DeepSea+](https://github.com/turningpoint1988/NLDNN), [DanQ+](https://github.com/turningpoint1988/NLDNN), [DanQV+](https://github.com/turningpoint1988/NLDNN).
- Nucleotide-level models: [FCNA](https://github.com/turningpoint1988/FCNA), [BPNet](https://github.com/kundajelab/bpnet/), [Leopard](https://github.com/GuanLab/Leopard), [FCNA](https://github.com/turningpoint1988/FCNsignal).

## Data Preparation

- Download [hg38.fa](https://hgdownload.soe.ucsc.edu/downloads.html#human) and [mm10.fa](https://hgdownload.soe.ucsc.edu/downloads.html#mouse), and then put them into the `Genome` directory.
- Download [TF binding datasets](https://www.encodeproject.org) and rename peak files as 'ChIPseq.${Species}.${Cell.TF}.idr.bed' and coverage track files as 'ChIPseq.${Species}.${Cell.TF}.pv.bigWig', where ${Species} denotes the species of Human or Mouse and ${Cell.TF} denotes the name of a cell-specific TF, and then put them into the `Human-Mouse` directory.
- Download [Chromatin accessibility datasets](https://www.encodeproject.org) and rename coverage files as '${Species}.${Cell}.chromatin.fc.bigWig', where ${Species} denotes the species of Human or Mouse and ${Cell} denote the name of a cell type, and then put them into the `Chromatin` directory.
- Three types of SNPs are already involved in this repository, pls refer to the `SNP` directory.

After these are finished, you can run the following shell script to prepare TF binding data.

```
bash annotate.sh
```

By doing this, DNA sequences for each cell-specific TF will be divided into the test (chr1,chr18), validation (chr8), and training (the remaining chromosomes except chrY) sets, in which all TF binding peaks (600bp) are regarded as positive sequences while sequences (600bp) that do not overlap with positive sequences and match the GC distribution of positive ones are regarded as negative sequences. 


## Stage 1: Training NLDNN on the source species (Fig.1d)

The stage is to train NLDNN using the training set from the source species, and then evaluate NLDNN using the test set from the source or target species.

```
bash run.sh
```

This execution includes a ‘warm-up’ process to select the best-initialized model, then the selected model is used as an initialized template for the training phase. 

## Stage 2: Fine-tune NLDNN by adversarial training in a dual-path framework (Fig.1e)

This stage is performed in a dual-path framework where the top path contains a source generator for generating feature mappings from the source species while the bottom path contains a target generator for generating feature mappings from the target species, on top of which a discriminator is appended to discriminate the source and target species.

## Stage 3: Test NLDNN on the target species (Fig.1f)

After adversarial training, the target generator and predictor were concatenated to predict the coverage values of the test set from the target species.. 

```
bash run_at.sh
```

Note that the above shell script includes stages 1 and 2.

## Predictive performance

To compare the predictive performance of NLDNN with the competing methods, PR-AUC and Pearson correlation were adopted to separately assess the classification and fitting performance of them. The comparison scatter plot is shown as follows:

<p align="center"> 
<img src=https://github.com/turningpoint1988/NLDNN/blob/main/Pictures/predictive.jpg width = "600" height = "400">
</p>

<h4 align="center"> 
Fig.2 Within- and cross-species performance comparison between all models for predicting TF binding.
</h4>

## Motif Prediction

Motif prediction on the specified test data:

```
python motif_prediction.py -d <> -n <> -g <> -t <> -c <> -o <>
```

| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/FCNsignal/HeLa-S3/CTCF/data                |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -t         | The threshold value (default is 0.3)                                                        |
| -c         | The trained model path of a specified dataset, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF|
| -o         | The path of storing motif files, e.g. /your_path/FCNsignal/motifs/HeLa-S3/CTCF              |

### Output

Generate motif files in MEME format, which are subsequently used by TOMTOM.


## Locating TFBSs

Locating potential binding regions on inputs of arbitrary length:

```
python TFBS_locating.py -i <> -n <> -g <> -t <> -w <> -c <>
```
| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -i         | The input file in bed format, e.g. /your_path/FCNsignal/input.bed                           |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -t         | The threshold value to determine the binding regions (default is 1.5)                       |
| -w         | The length of the binding regions (default is 60)                                           |
| -c         | The trained model path of a specified dataset, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF|

### Output

The outputs include the base-resolution prediction of inputs and the position of potential binding regions in the genome (bed format). <br/>
We also provide the line plots of the above base-resolutiion prediction. For example:

<p align="center"> 
<img src=https://github.com/turningpoint1988/FCNsignal/blob/main/output.jpg>
</p>
