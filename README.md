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

You can download related ChIP-seq TF binding datasets using the following shell script:

```
bash download.sh
```

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

Note that the above shell script includes stages 2 and 3.

## Predictive performance

To compare the predictive performance of NLDNN with the competing methods, PR-AUC and Pearson correlation are adopted to separately assess the classification and fitting performance of them. The comparison scatter plot is shown as follows:

<p align="center"> 
<img src=https://github.com/turningpoint1988/NLDNN/blob/main/Pictures/predictive.jpg width = "600" height = "500">
</p>

<h4 align="center"> 
Fig.2 Within- and cross-species performance comparison between all models for predicting TF binding.
</h4>

## Variant effect prediction

To assess the performance of NLDNN and the competing methods on variant effect prediction, three different types of tasks are constructed: classification, regression, and prioritization. All related SNPs (including TF-specific SNPs, MPRA, causal SNPs) are provided in the `SNP` directory.

- **Task1: TF-specific SNPs classification.** 

Models are trained using ChIP-seq TF binding datasets, and then used to evaluate their performance on the task by the following shell script:

```
cd SNP
bash predict.sh
```

- **Task2: MPRA regression.** 

Models are trained using related chromatin accessibility data, and then used to evaluate their performance on the task by the following python script:

```
cd SNP
python test_mpra.py -r <> -m <>
```

| Arguments  | Description                                                               |
| ---------- | -----------------------------------------------------------------------   |
| -r         | The path of the project, e.g., ${HOME}/NLDNN                              |
| -m         | The name of saved models on a specific cell, e.g. models_NLDNN_GM12878    |

- **Task3: Causal SNPs prioritization.** 

Models are trained using chromatin accessibility data (GM12878), and then used to evaluate their performance on the task by the following python script:

```
cd SNP
python test_causal.py -r <> -m <>
```

| Arguments  | Description                                                               |
| ---------- | -----------------------------------------------------------------------   |
| -r         | The path of the project, e.g., ${HOME}/NLDNN                              |
| -m         | The name of saved models on a specific cell, e.g. models_NLDNN_GM12878    |

## Visualization of the contribution scores of SNPs

Firstly, the contribution scores of SNPs are computed using the following shell script:

```
cd SNP
bash contrib.sh
```
Secondly, the predictions and contributions of each SNP (ref and alt sequences) are visualized through the following shell script:

```
cd SNP
bash visualization.sh
```

An example is shown as follows:

<p align="center"> 
<img src=https://github.com/turningpoint1988/NLDNN/blob/main/Pictures/SNP.jpg>
</p>


## Localization ability

Given that nucleotide-level models all have the ability to locate potential TF binding regions, so only nucleotide-level models are compared. <br/>
Firstly, Chromosome 1 is segmented into non-overlapping windows of 600bp using the following shell script:

```
cd chr1
bash make_windows.sh
```

Secondly, DNA sequences are encoded into one-hot matrics, and trained models take one-hot matrics as input and output their predicted coverage values. Through the maximums of the predicted coverage values, we can exactly know their positions on chromosomes.
To evaluate the performance of nucleotide-level models for locating TF binding regions, a direct way and an indirect way are used.

```
cd chr1
bash locating.sh
```

## Visualization of outputs

Although NLDNN is trained using DNA sequences of length 600bp, it can accept inputs of arbitrary length and predict each nucleotide’s value. Therefore, we segment chromosome 1 into non-overlapping regions of length 100kb <br/>
- **Visualization of contributions.**  The contributions and ISSM values of each peak in these regions could be visualized using the following shell script.
- **Visualization of predictions.**  The true coverage values of these regions (100kb) and corresponding predicted values obtained through human, mouse and mouse-adaptation models could be visualized using the same script.

```
bash plot.sh
```

An example of the contributions and ISSM values of a peak from a human genomic region (chr1:117600000-117700000, 100kb) is displayed as follows:

<p align="center"> 
<img src=https://github.com/turningpoint1988/NLDNN/blob/main/Pictures/Contrib.ISSM.jpg>
</p>


An example of predictions of a human genomic region (chr1:15600000-15700000, 100kb) is displayed as follows:

<p align="center"> 
<img src=https://github.com/turningpoint1988/NLDNN/blob/main/Pictures/prediction.jpg>
</p>
