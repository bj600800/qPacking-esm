# qPacking
Investigation of hydrophobic stacking patterns in protein folding superfamilies based on the TED database, leveraging the ESM2-650M language model.

**Content**
```
├── experiments/         # Experiment configuration files (e.g., YAML)
├── logs/                # Log output directory (e.g., train.log, test.log)
├── predict.py           # Inference script entry point
├── scripts/             # Execution scripts: training, testing, zero-shot inference
├── README.md            # Project documentation
└── qpacking/            # Main project code package
    ├── config.py            # Configuration loading and parsing
    ├── data_prepare/        # Data preprocessing utilities
    ├── hydrocluster/        # Hydrophobic cluster identification and analysis
    ├── models/              # Core DL modules: model, dataset, training, evaluation
    └── utils/               # Utility functions: logging, visualization, etc.
```

**Dependence:**
```
python=3.10
pip install tqdm
pip install fair-esm (version=2.0.0)
pip install melodia-py (version=0.1.4)
```

**Download checkpoints:**
ESM2-weight: https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

ESM2-contact-regression: https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
**Tools:**
```
conda install -c conda-forge -c bioconda mmseqs2
conda install -c bioconda seqkit
conda install bioconda::cd-hit
pip install pdb-tools
conda install salilab::dssp (version=3.0.0)
conda install libboost==1.73.0
```

## Notice
- Checkpoints and used-datasets are available in the zenodo.