# qPacking-ESM
Structure-aware fine-tuning strategies for ESM-2 to improve protein fitness prediction and explainability

## Installation qPacking-ESM
```
python=3.10.13
(GPU) conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
(CPU) conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch
pip install sympy==1.13.1
pip install transformers==4.52.4
pip install datasets
pip install peft
pip install matplotlib==3.10.7
pip install biopython
pip install scikit-learn
pip install mlflow
pip install melodia-py==0.1.4
pip install umap-learn
pip install pytest
pip install colorlog
pip install biotite
pip install numpy==2.2.6

Tools:
conda install bioconda::tmalign
conda install -c conda-forge -c bioconda foldseek
conda install -c conda-forge -c bioconda mmseqs2
conda install -c bioconda seqkit
conda install bioconda::cd-hit
pip install pdb-tools
conda install salilab::dssp (version=3.0.0)
conda install libboost==1.73.0
```

## Train qPacking-ESM

  ```
  python train_qpacking_esm.py --yaml [config file path in train_configs dir] 
  ```

## Download ESM2 model on Huggingface

  ```
  https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main
  ```