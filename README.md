# qPacking
Exploring hydrophobic stacking patterns in protein folding superfamilies from TED database, laveraging ESMC-600M model.

**Dependance:**
```
python=3.19
pip install tqdm
pip install esm
pip install melodia-py (version=0.1.4)
```

**Tools:**
```
conda install -c conda-forge -c bioconda mmseqs2
conda install -c bioconda seqkit
conda install bioconda::cd-hit
pip install pdb-tools
conda install salilab::dssp (version=3.0.0)
conda install libboost==1.73.0
```

**Directory function:**
- checkpoints: Contains the checkpoints for the model.
- data: Contains the data files.
- experiments: different experimental configurations for experiment management and reproducibility.
- logs: Contains the logs for the experiments.
- qpacking: Contains the main codes for the project.
  - data_prepare: Contains the codes for data preparation.
  - dataset: Contains the codes for dataset preparation.
  - hydrocluster: hydrophobic cluster calculation
  - models: Contains the codes for the models.
  - scripts: train, test, and evaluation scripts.
  - utils: Contains the utility codes for reuse.