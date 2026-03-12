<div align="center">
<h1>Synthetic Data Generation for Brain-Computer Interfaces</h1>
<h3>Overview, Benchmarking, and Future Directions</h3>

[Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1</sup>, Zhentao He, [Xingyi He](https://github.com/BAY040210)<sup>1</sup>, [Hongbin Wang](https://github.com/WangHongbinary)<sup>1</sup>, [Tianwang Jia](https://github.com/TianwangJia)<sup>1</sup>, Jingwei Luo, [Siyang Li](https://scholar.google.com/citations?user=5GFZxIkAAAAJ&hl=en)<sup>1</sup>, [Xiaoqing chen](https://scholar.google.com/citations?user=LjfCH7cAAAAJ&hl=en), and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 :email:</sup>

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

(<sup>:email:</sup>) Corresponding Author

</div>

This repository contains the source code for our paper **Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions**.

This survey provides a comprehensive review of brain signal generation for BCIs, covering methodological taxonomies, benchmark experiments, evaluation metrics, and key applications. We systematically categorize existing generative algorithms into four types: knowledge-based, feature-based, model-based, and translation-based approaches. Furthermore, we benchmark existing brain signal generation approaches across four representative BCI paradigms to provide an objective performance comparison. Finally, we discuss the potentials and challenges of current generation approaches and prospect future research on accurate, data-efficient, and privacy-aware BCI systems. 

## Data scarcity in BCI
Recent success of deep learning underscores the importance of large, high-quality training data. For example, Google’s trillion-word corpus has significantly boosted performance in language models. However, acquiring sufficient brain signals presents challenges.

<p align="center"><img width="394" height="426" alt="image" src="https://github.com/user-attachments/assets/018b5b9b-6fe0-45b5-8a9f-b29d6568ab5c" /></p>

<p align="center"><font color="gray">Figure 1: Data scarcity issue in BCI.</font></p>

## Framework
Data generation driven machine learning pipeline for BCIs, which includes brain signal acquisition, data preprocessing, data generation, feature engineering, and classification/regression. The latter two components can be unified into a single end-to-end neural network. Data generation approaches are categorized into four types: (a) knowledge-based generation, (b) feature-based generation, (c) model-based generation, and (d) translation-based generation.

<img width="1075" height="639" alt="image" src="https://github.com/user-attachments/assets/b0793e9f-541d-45eb-a58b-61c841d4367f" />

<p align="center"><font color="gray">Figure 2: Data generation driven machine learning pipeline for BCIs.</font></p>


## Code Structure

The repository is organized into two main components: **synthetic brain signal generation** and **synthetic data evaluation**.

```
.
├── Synthetic_Data_Evaluation/ # Evaluation metrics for synthetic EEG data
│ ├── FID.py # Fréchet Inception Distance
│ ├── MMD.py # Maximum Mean Discrepancy
│ ├── SignalToNoiseRatio.py # Signal-to-noise ratio metrics
│ ├── CosineSimilarity.py # Cosine similarity
│ ├── JaccardSimilarity.py # Jaccard similarity
│ ├── InceptionScore.py # Inception Score
│ └── EntropyStats.py # Entropy and diversity statistics
│
├── Synthetic_Data_Generation/ # Brain signal generation methods
│
│ ├── Knowledge_based/ # Knowledge-based augmentation methods
│ │ ├── Noise.py # Adding uniform noise to an EEG trial in the time domain
│ │ ├── Scale.py # Scaling the voltage of EEG trials with a minor coefficient
│ │ ├── Flip.py # Performing voltage inversion
│ │ ├── FShift.py # Shifting the frequency of EEG trials
│ │ ├── FSurr.py # Replacing the Fourier phases of trials with random numbers
│ │ ├── CR.py # Swapping symmetric left and right hemisphere channels
│ │ ├── HS.py # Recombining left and right channels from the same category
│ │ ├── DWTAug.py # Decomposing trials by DWT and reassembling coefficients
│ │ ├── HHTAug.py # Empirical mode decomposition and sub-signal reassembling
│ │ └── utils/ # Utility functions (alignment, splitting, etc.)
│
│ └── Model_based/ # Deep generative models
│ └── SSVEP_AUG/
│ ├── etc/ # Experiment configuration
│ │ ├── config.yaml
│ │ └── global_config.py
│ │
│ ├── Utils/ # Data processing, training, and evaluation utilities
│ │ ├── dataprocess.py
│ │ ├── EEGDataset.py
│ │ ├── Trainer.py
│ │ └── test.py
│ │
│ └── Models/ # Neural network architectures
│ ├── Generator.py # GAN/VAE generators
│ ├── Discriminator.py # Discriminators
│ ├── VAE.py # Variational autoencoders
│ │
│ ├── KNoW/ # Classical BCI algorithms (CCA, TRCA, FBCCA, TDCA)
│ ├── DeepL/ # Deep learning decoders (EEGNet, SSVEPformer, etc.)
│ └── HZTKD/ # Filter-bank and knowledge-based layers
│
└── main.py # Entry point for training and evaluation
```

---

## Installation

```bash
git clone https://github.com/wzwvv/DG4BCI.git
cd DG4BCI
pip install -r requirements.txt
```
