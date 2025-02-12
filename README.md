
## Introduction

R-loops are three-stranded nucleic acid structures consisting of an RNA–DNA hybrid and a displaced single-stranded DNA that frequently occur during transcription from bacteria to mammals. Current methods for R-loop sites prediction are based on DNA sequence, which precludes detection across cell types, tissues or developmental stages. Here we provide _virtuRloops_, a computational tool that allows the prediction of cell type specific R-loops using sequence information and high throughput sequencing datasets. Human and mouse predictions can be downloaded and visualized in the [*following link*](http://193.147.188.155/pmargar/rloops_pred/). 

## Features

Multiple combinations of features were tested to generate our training models, which are deposited in models directory. Full models were trained with 3 sequence-based features and 9 high throughput sequencing signals, namely H3K9ac, H3K9me3, H3K27me3, H3K4me3, H3K36me3, H3K27ac, H3K4me1, CTCF, RAD21, RNA-seq, DNase-seq and GRO-seq. Sequence-based features constist of G+C content and the outputs of [*deepRloopPre*](https://github.com/PEHGP/deepRloopPre) and [*QmRLFS-finder*](https://github.com/piroonj/QmRLFS-finder). We found that specific combinations of the most informative features were able to produce accurate predictions (see performance directory), so there is no need to provide all the above datasets to obtain reliable outcomes. We encourage the user to check the different performances and look for the model that best suits his/her datasets.

## Usage

One of the aforementioned models should be passed to predict R-loop sites, together with a tabulated file containing the paths of the corresponding feature files (in 4 columns bedgraph format; see features directory). Note that the program assumes that feature files are sorted and RPM normalized. If a target peak file (sorted and in 3 columns bed format) is provided, predictions will be performed on such coordinates. Peaks will be resized to 500 bp if they don't have such size. If no peak file is provided, whole genome predictions will be computed. For assemblies other than mm9 or hg38, a genome fasta file should be provided. In such case, in order to include _deepRloopPre_ as a predictive feature (recommendable), unstranded _deepRloopPre_ scores should be provided both as a bedgraph file and also in the feature table (feature name = deepRloop).

```
Usage: ./predict [options]

Options:
	-h, --help
		Show this help message and exit

        -n CHARACTER, --name=CHARACTER
                Execution name (predictions will be reported in a file with this name)

        -d CHARACTER, --datasets=CHARACTER
                Tabulated file (with no header line) containing paths of sequencing datasets (RPM normalized and 4-columns sorted bedgraph format).
                Note that model features should be provided and others will be ignored.
                Possible feature names are: H3K9ac,H3K9me3,H3K27me3,H3K4me3,H3K36me3,H3K27ac,H3K4me1,CTCF,RAD21,RNAseq,DNase,GROseq.
                If assembly is different than 'mm9' or 'hg38', unstranded deepRloopPre scores should be also included both as a
                bedgraph format and also in the feature table (feature name = deepRloop)

        -t CHARACTER, --target=CHARACTER
                Input bed file (3-columns, sorted and with no header). R-loop model will be applied to supplied loci.
                If no file is given, whole genome predictions will be generated

        -a CHARACTER, --assembly=CHARACTER
                Genome assembly. If other than 'mm9' or 'hg38', a fasta file with genome sequences should be provided

        -f CHARACTER, --fasta=CHARACTER
                Genome fasta file. Only required if assembly is different than 'mm9' or 'hg38'

        -m CHARACTER, --model=CHARACTER
                R-loop model file in RData format (check 'models' directory)

        -h, --help
                Show this help message and exit
```
Calling example for given peak file:
```
./predict -n example1 -d features/features_adrenal_gland_chr22.csv -t peaks/peaks_example_chr22.bed -m models/deepRloop_GC_RLFS_DNase_RNA_K4me3_K9me3_K36_me1/model.RData
```
Calling example for whole genome prediction (needs a user provided features table):
```
./predict -n example2 -d myFeatures.csv -t hg38 -m models/deepRloop_GC_RLFS_DNase_RNA_K4me3_K9me3_K36_me1/model.RData
```

## Requirements

- [bedtools v2.26.0](https://bedtools.readthedocs.io/en/latest/)
- [samtools v1.3](https://github.com/samtools/samtools)
- [R >= 3.6](https://cran.r-project.org/)

### R libraries:

- [optparse](https://cran.r-project.org/web/packages/optparse/index.html)
- [stringi](https://cran.r-project.org/web/packages/stringi/index.html)
- [randomForest](https://cran.r-project.org/web/packages/randomForest/index.html)
