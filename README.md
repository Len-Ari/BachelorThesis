# Repo Overview
This is the repository for all tests and evaluations of my Bachelorsthesis.

## Packages

All packages necessary are in the `environment.yaml` file, which can be used to create a conda environment.
Additionally, BeIR needs to be installed using `pip install beir`...

## Missing Folders

Since some parts of the project are not mine and instead models from huggingface or data from Pubmed, the folders containing these parts are not in this repository.
If any results are to be recreated the following folders need to be added and filled:

### Models

In this directory all models need to be saved. The models that were evaluated in the thesis can all be downloaded using `Load_model.ipynb`.
Based on this the fine-tuned models can also be created by using the files  `TrainTransformerv1.ipynb`, `TrainTransformerv2.ipynb` and `train_beir.py`.

### Data

In the data folder the Pubmed articles need to be added.
They can be extracted from https://pmc.ncbi.nlm.nih.gov/tools/openftlist/ using their official tools.

For this the structure descibed in the thesis needs to be rebuild. Additionally the three datasets as they are shown in the thesis need to be created.

## Plots
All plots that are shown in the Thesis, as well as some additional ones, are saved in the Plots folder.
