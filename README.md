# Welcome to PhaseNet: A computational framework for identifying phase separation proteins based on protein language model

Biomolecular condensates formed through liquid–liquid phase separation (LLPS) are essential for a wide range of cellular
functions, and their dysregulation is implicated in various diseases. Accurate identification of phase separation proteins is
therefore critical for understanding the molecular mechanisms underlying these condensates. Here, we propose PhaseNet,
a dual-task computational framework designed to simultaneously distinguish LLPS proteins from non-LLPS proteins
and further classify LLPS proteins into PS-Self and PS-Part. In the first task, PhaseNet leverages a combination of
pretrained protein language models (ESM), sequence encodings (ZSCALE and BLOSUM), and a modular architecture
incorporating CNN-BiGRU and multi-head attention mechanisms. These heterogeneous features are integrated through
attention-guided fusion and optimized using an HSIC-based regularization strategy to enhance discriminative capacity.
In the second task, PhaseNet applies Lasso-based feature selection on ESM-derived embeddings, followed by a stacking
ensemble composed of five classifiers, including Random Forest, Extra Trees, GBDT, XGBoost, and MLP. Extensive
benchmarking across multiple independent test sets demonstrates that PhaseNet outperforms existing LLPS predictors
in both general identification and fine-grained classification tasks. This modular and interpretable framework provides a
robust tool for the systematic discovery and annotation of phase separation proteins.

![The workflow of this study](https://github.com/SamHe6/PhaseNet/blob/main/workflow.png)

# Dataset for this study
We provided our dataset and you can find them [Dataset](https://github.com/SamHe6/PhaseNet/tree/main/Dataset)
# Source code
We provide the source code and you can find them [Code]()
