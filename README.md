
## Introduction

R-loops are three-stranded nucleic acid structures consisting of an RNA–DNA hybrid and a displaced single-stranded DNA that frequently occur during transcription from bacteria to mammals. Current methods for R-loop site prediction are based on DNA sequence, which precludes detection across cell types, tissues or developmental stages. Here we provide virtuRloops, a computational tool that allows the prediction of cell type specific R-loops using genomic features. Human and mouse predictions can be downloaded and visualized in the [*following link*](http://193.147.188.155/pmargar/drip_pred/). 

## Usage

virtuRloops provides pre-computed learning models that were trained using multiple combinations of sequencing features (see models directory). One of these model should be passed to predict R-loop sites, together with a tabulated file containing the paths of the corresponding feature files (in 4 columns bedgraph format; see features directory). Note that the program assumes that feature files are RPM normalized. If a target peak file (in 3 columns bed format) is provided, predictions will be performed on such coordinates. Peaks will be resized to 500 bp if they don't have such size. If no peak file is provided, either the mouse (mm9) or human (hg38) genomes will be used as target. Note that, since GitHub limits the size of files, we do not directly provide the corresponding whole genome bed files for these species. To compute whole genome predictions, download the file you are interested in from the following links and put it on the 'genomes' folder: [*hg38*](http://193.147.188.155/pmargar/drip_pred/hg38_1_to_22_XYM_sliding_500bp_MAPPABLE.bed), [*mm9*](http://193.147.188.155/pmargar/drip_pred/mm9_20_longest_sliding_500bp_MAPPABLE.bed).

```
Usage: ./predict [options]

Options:
	-n CHARACTER, --name=CHARACTER
		Execution name (predictions will be reported in a file with this name)

	-d CHARACTER, --datasets=CHARACTER
		Tabulated file (with no header line) containing paths of sequencing datasets (RPM normalized and 4-columns bedgraph format; check 'features' directory).
		Note that model features should be provided and others will be ignored.
		Possible feature names are: H3K9ac,H3K9me3,H3K27me3,H3K4me3,H3K36me3,H3K27ac,H3K4me1,RNA_seq,DNase,GRO_seq

	-t CHARACTER, --target=CHARACTER
		Genome assembly name or input bed file with no header. If bed file is supplied, R-loop model will be applied to its corresponding loci.
		Otherwise, genome wide predictions will be generated (only 'mm9' and 'hg38' are supported)

	-m CHARACTER, --model=CHARACTER
		R-loop model file in RData format (check 'models' directory)

	-h, --help
		Show this help message and exit
```
Calling example for given peak file:
```
./predict -n example1 -d features/features_adrenal_gland_chr22.csv -t peaks/peaks_example_chr22.bed -m models/DRIP_E14_3T3_comb/RNA_DNase_K4me3_K9me3_K36_K4me1/model.RData
```
Calling example for whole genome prediction (needs a user provided features table):
```
./predict -n example2 -d myFeatures.csv -t hg38 -m models/DRIP_E14_3T3_comb/RNA_DNase_K4me3_K9me3_K36_K4me1/model.RData
```

## Requirements

- [bedtools v2.26.0](https://bedtools.readthedocs.io/en/latest/)
- [R >= 3.6](https://cran.r-project.org/)

### R libraries:

- [optparse](https://cran.r-project.org/web/packages/optparse/index.html)
- [stringi](https://cran.r-project.org/web/packages/stringi/index.html)
- [randomForest](https://cran.r-project.org/web/packages/randomForest/index.html)
