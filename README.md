# qPacking
Structure-aware fine-tuning strategies for ESM-2 to improve protein fitness prediction and explainability

## Installation qPacking
```
python=3.10
(GPU) conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
(CPU) conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch
pip install transformers==4.52.4
pip install sympy==1.13.1
pip install datasets
pip install matplotlib
pip install biopython
pip install peft
pip install scikit-learn
pip install mlflow
pip install tqdm
pip install melodia-py (version=0.1.4)
pip install umap-learn

conda install -c conda-forge -c bioconda foldseek
conda install -c conda-forge -c bioconda mmseqs2
conda install -c bioconda seqkit
conda install bioconda::cd-hit
pip install pdb-tools
conda install salilab::dssp (version=3.0.0)
conda install libboost==1.73.0
```
## Run qPacking


## Notice
Publish as preprint in bioRxiv.