# qPacking
Structure-informed instruct fine-tuning strategies for ESM-2 to improve 
protein representations and enable interpretable thermostability effect prediction

**Content**

All params stored in experiment.yaml from experiments/. Config.py manages these yamls, and passes the params to predict.py. 
```
qPacking/
├── logs/                            # Training output directory for logs
│
├── predict.py                       # Entry point for inference (model prediction)
│
├── train.py                         # Training entry script (task-dependent launcher)
│
├── README.md                        # Project documentation
│
├── qpacking/                        # Main source code package
│   ├── data_prepare/                # Data preprocessing utilities (download, clean, convert, etc.)
│   ├── hydrocluster/                # Hydrophobic cluster identification and feature analysis
│   ├── models/                      # Core deep learning modules (model, dataset, training, evaluation)
│   └── utils/                       # Utility functions (logging, visualization, configuration helpers)
│
├── scripts/                         # Execution scripts (training, evaluation, inference)
│   └── configs/                     # Centralized management of experiment hyperparameters
│       ├── hydrophobic_binary.yaml  # Hyperparameter file for hydrophobic binary classification task
│       └── config.py                # Decouples data flow from code flow: loads and validates parameters
```

**Dependence:**
```
python=3.10
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers
pip install sympy==1.13.1
pip install datasets
pip install matplotlib
pip install biopython
pip install peft
pip install scikit-learn
pip install mlflow
pip install tqdm
pip install fair-esm (version=2.0.0)
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

**Download checkpoints:**

ESM2-weight: https://huggingface.co/facebook/esm2_t36_3B_UR50D

ESM2-contact-regression: https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt

## Notice
- Checkpoints and used-datasets are available in zenodo.