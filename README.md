<div align="center">
<h1>Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions</h1>

[Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1</sup>, Zhentao He<sup>1</sup>, [Xingyi He](https://github.com/BAY040210)<sup>1</sup>, [Hongbin Wang](https://github.com/WangHongbinary)<sup>1</sup>, [Tianwang Jia](https://github.com/TianwangJia)<sup>1</sup>, Jingwei Luo<sup>1</sup>, [Siyang Li](https://scholar.google.com/citations?user=5GFZxIkAAAAJ&hl=en)<sup>1</sup>, [Xiaoqing chen](https://scholar.google.com/citations?user=LjfCH7cAAAAJ&hl=en)<sup>1 2</sup>, and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 2 :email:</sup>

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

<sup>2</sup> Zhongguancun Academy

(<sup>:email:</sup>) Corresponding Author

</div>

This repository contains the source code for our paper **Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions**.

This survey provides a comprehensive review of brain signal generation for BCIs, covering methodological taxonomies, benchmark experiments, evaluation metrics, and key applications. We systematically categorize existing generative algorithms into four types: knowledge-based, feature-based, model-based, and translation-based approaches. Furthermore, we benchmark existing brain signal generation approaches across four representative BCI paradigms to provide an objective performance comparison. Finally, we discuss the potentials and challenges of current generation approaches and prospect future research on accurate, data-efficient, and privacy-aware BCI systems. 

## Research Background
Recent success of deep learning underscores the importance of large, high-quality training data. For example, Google’s trillion-word corpus has significantly boosted performance in language models. However, acquiring sufficient brain signals presents challenges.

<p align="center"><img width="492.5" height="532.5" alt="image" src="https://github.com/user-attachments/assets/018b5b9b-6fe0-45b5-8a9f-b29d6568ab5c" /></p>

<p align="center"><font color="gray">Figure 1: Data scarcity issue in BCI.</font></p>

## Synthetic Brain Signal Generation
Data generation driven machine learning pipeline for BCIs, which includes brain signal acquisition, data preprocessing, data generation, feature engineering, and classification/regression. The latter two components can be unified into a single end-to-end neural network. Data generation approaches are categorized into four types: (a) knowledge-based generation, (b) feature-based generation, (c) model-based generation, and (d) translation-based generation.

<p align="center"><img width="1075" height="639" alt="image" src="https://github.com/user-attachments/assets/b0793e9f-541d-45eb-a58b-61c841d4367f" /></p>

<p align="center"><font color="gray">Figure 2: Data generation driven machine learning pipeline for BCIs.</font></p>

### Knowledge-Based Generation
Knowledge-based brain signal generation leverages estab- lished neurophysiological priors, such as event-related desyn- chronization and synchronization patterns in MI and rhythmic spike-wave discharges in epilepsy, to guide the generation of synthetic data.

<p align="center"><img width="438" height="391" alt="image" src="https://github.com/user-attachments/assets/f7016a58-d096-48a8-9cb0-db71f0c7de16" /></p>

<p align="center"><font color="gray">Figure 3: Visualizations of brain signals before (blue lines) and after (red lines) eleven knowledge-based generation approaches, using 1-channel as an example.</font></p>

### Model-Based Generation
Model-based generation is more flexible, commonly relying on probabilistic generative models to generate synthetic brain signals, based on the idea that the underlying distribution of brain signals can be modeled and sampled from sophisticated deep learning models.

<p align="center"><img width="551" height="574" alt="image" src="https://github.com/user-attachments/assets/9f1c8ed9-10b3-4ff0-a8a8-0d399927598e" /></p>

<p align="center"><font color="gray">Figure 4: Model-based generation approaches for brain signals, including GANS, VAEs, AMs, and DDPMs.</font></p>

### Translation-Based Generation
Translation-based generation involves synthesizing data by integrating information from additional modalities, typically through cross-modal generative models. In BCIs, this strategy is crucial for bridging diverse data types, such as brain signals, text, speech, images, and other sensor data.

<p align="center"><img width="1129" height="410" alt="image" src="https://github.com/user-attachments/assets/b1e17320-966d-4995-9b03-7d4d9a3c238b" /></p>

<p align="center"><font color="gray">Figure 5: Two types of translation-based generation for brain signals, taking two modalities of image and brain signal as an example.</font></p>

## Synthetic Brain Signal Evaluation
The evaluation of generated data is pivotal for data- centric artificial intelligence. A well-defined and scientifically grounded evaluation framework is essential for guiding data generation and utilization. This work introduces an evaluation framework to assess the synthetic data across six dimensions: reliability, quality, task performance, model performance, multimodal consis- tency, and privacy preservation ability.

<p align="center"><img width="1127" height="640" alt="image" src="https://github.com/user-attachments/assets/d7945d5e-bf44-4c0b-82eb-bdf54ae88e0d" /></p>

<p align="center"><font color="gray">Figure 6: Evaluation framework of generated data in BCIs.</font></p>

## Code Structure

The repository is organized into two main components: **synthetic brain signal generation** and **synthetic data evaluation**.

```
.
├── Synthetic_Data_Evaluation/ # Evaluation metrics for synthetic EEG data
│ ├── FID.py # Fréchet Inception Distance
│ ├── MMD.py # Maximum Mean Discrepancy
│ ├── SNR.py # Signal-to-noise ratio metrics
│ ├── CosineSim.py # Cosine similarity
│ ├── JaccardSim.py # Jaccard similarity
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
