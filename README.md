# qPacking
Exploring hydrophobic stacking patterns in protein folding superfamilies from TED database, laveraging ESMC-600M model.

**Dependence:**
```
python=3.10
pip install tqdm
pip install fair-esm
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
- data: Contains the data files.
- experiments: different experimental configurations for experiment management and reproducibility.
- logs: Contains the logs for the experiments.
- qpacking: Contains the main codes for the project.
  - data_prepare: Contains the codes for data preparation.
  - dataset: Contains the codes for dataset preparation.
  - hydrocluster: hydrophobic cluster calculation
  - models: Contains the codes for ESM fine-tuning.
  - scripts: zero-shot prediction, train, test, and evaluation.
  - utils: Utility codes for reuse.

## Notice
- Checkpoint files are available in the zenodo.
- Fine-tuning data are available in the zenodo.