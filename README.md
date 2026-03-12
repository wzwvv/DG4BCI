# Synthetic Data Generation for Brain-Computer Interfaces

### Overview, Benchmarking, and Future Directions

[Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)1, Zhentao He, [Xingyi He](https://github.com/BAY040210)1, [Hongbin Wang](https://github.com/WangHongbinary)1, [Tianwang Jia](https://github.com/TianwangJia)1, Jingwei Luo, [Siyang Li](https://scholar.google.com/citations?user=5GFZxIkAAAAJ&hl=en)1, [Xiaoqing chen](https://scholar.google.com/citations?user=LjfCH7cAAAAJ&hl=en), and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)1 :email:

1 School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

(:email:) Corresponding Author



This repository contains the source code for our paper **Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions**.

This survey provides a comprehensive review of brain signal generation for BCIs, covering methodological taxonomies, benchmark experiments, evaluation metrics, and key applications. We systematically categorize existing generative algorithms into four types: knowledge-based, feature-based, model-based, and translation-based approaches. Furthermore, we benchmark existing brain signal generation approaches across four representative BCI paradigms to provide an objective performance comparison. Finally, we discuss the potentials and challenges of current generation approaches and prospect future research on accurate, data-efficient, and privacy-aware BCI systems. 

## Data scarcity in BCI

Recent success of deep learning underscores the importance of large, high-quality training data. For example, Google’s trillion-word corpus has significantly boosted performance in language models. However, acquiring sufficient brain signals presents challenges.



Figure 1: Data scarcity issue in BCI.

## Synthetic Brain Signal Generation

Data generation driven machine learning pipeline for BCIs, which includes brain signal acquisition, data preprocessing, data generation, feature engineering, and classification/regression. The latter two components can be unified into a single end-to-end neural network. Data generation approaches are categorized into four types: (a) knowledge-based generation, (b) feature-based generation, (c) model-based generation, and (d) translation-based generation.



Figure 2: Data generation driven machine learning pipeline for BCIs.

### Knowledge-Based Generation

Knowledge-based brain signal generation leverages estab- lished neurophysiological priors, such as event-related desyn- chronization and synchronization patterns in MI and rhythmic spike-wave discharges in epilepsy, to guide the generation of synthetic data.



Figure 3: Visualizations of brain signals before (blue lines) and after (red lines) eleven knowledge-based generation approaches, using 1-channel as an example.

### Model-Based Generation

Model-based generation is more flexible, commonly relying on probabilistic generative models to generate synthetic brain signals, based on the idea that the underlying distribution of brain signals can be modeled and sampled from sophisticated deep learning models.



Figure 4: Model-based generation approaches for brain signals, including GANS, VAEs, AMs, and DDPMs.

### Translation-Based Generation

Translation-based generation involves synthesizing data by integrating information from additional modalities, typically through cross-modal generative models. In BCIs, this strategy is crucial for bridging diverse data types, such as brain signals, text, speech, images, and other sensor data.



Figure 5: Two types of translation-based generation for brain signals, taking two modalities of image and brain signal as an example.

## Synthetic Brain Signal Evaluation

The evaluation of generated data is pivotal for data- centric artificial intelligence. A well-defined and scientifically grounded evaluation framework is essential for guiding data generation and utilization. This work introduces an evaluation framework to assess the synthetic data across six dimensions: reliability, quality, task performance, model performance, multimodal consis- tency, and privacy preservation ability.



Figure 6: Evaluation framework of generated data in BCIs.

## Code Structure

The repository is organized into two main components: **synthetic brain signal generation** and **synthetic data evaluation**.

```
.
├── Synthetic_Data_Evaluation/ # Evaluation metrics for synthetic EEG data
│ ├── FID.py # Fréchet Inception Distance
│ ├── MMD.py # Maximum Mean Discrepancy
│ ├── SNR.py # Signal-to-noise ratio metrics
│ ├── CosineSimilarity.py # Cosine similarity
│ ├── JaccardSimilarity.py # Jaccard similarity
│ ├── IS.py # Inception Score
│ └── Entropy.py # Entropy and diversity statistics
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

