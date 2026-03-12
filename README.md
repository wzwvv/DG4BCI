# Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions
This repository contains the source code for our paper **Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions**.

This survey provides a comprehensive review of brain signal generation for BCIs, covering methodological taxonomies, benchmark experiments, evaluation metrics, and key applications. We systematically categorize existing generative algorithms into four types: knowledge-based, feature-based, model-based, and translation-based approaches. Furthermore, we benchmark existing brain signal generation approaches across four representative BCI paradigms to provide an objective performance comparison. Finally, we discuss the potentials and challenges of current generation approaches and prospect future research on accurate, data-efficient, and privacy-aware BCI systems. 

## Data scarcity in BCI
Recent success of deep learning underscores the importance of large, high-quality training data. For example, Google’s trillion-word corpus has significantly boosted performance in language models. However, acquiring sufficient brain signals presents challenges.

<p align="center"><img width="394" height="426" alt="image" src="https://github.com/user-attachments/assets/018b5b9b-6fe0-45b5-8a9f-b29d6568ab5c" /></p>

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

## Code structure

The following tree lists directories and files with short descriptions. For key modules in **Synthetic_Data_Generation**, main functions and classes are also listed.

---

**Synthetic_Data_Evaluation/** — *Evaluation metrics for generated data*

| File | Description |
|------|-------------|
| `FID.py` | Fréchet Inception Distance between real and generated features. |
| | `compute_fid()` — Compute FID; `_normalize_features_for_fid()` — Z-score normalization for FID. |
| `MMD.py` | Maximum Mean Discrepancy (RBF kernel). |
| | `compute_mmd()` — Compute MMD between real and gen distributions. |
| `SignalToNoiseRatio.py` | SNR in feature space or raw EEG space. |
| | `compute_snr_features()` — SNR in feature space (mean_std / power_ratio); `compute_snr_eeg()` — SNR in EEG space; `compute_snr_simple()` — Simple variance-based SNR in dB. |
| `CosineSimilarity.py` | Cosine similarity between real and generated features. |
| | `compute_cosine_similarity()` — Aggregate modes: mean_vectors, pairwise_mean; `compute_cosine_similarity_per_sample()` — Per-pair cosine (1-to-1). |
| `JaccardSimilarity.py` | Jaccard similarity on binarized features. |
| | `_binarize()` — Binarize features (median/mean/threshold); `compute_jaccard_similarity()` — Set-level Jaccard; `compute_jaccard_similarity_1to1()` — Per-sample pair Jaccard. |
| `InceptionScore.py` | Inception Score (classifier-based or histogram-based). |
| | `compute_inception_score()` — Train classifier on real, score on gen; `compute_inception_score_simple()` — Histogram-based diversity (no labels). |
| `EntropyStats.py` | Entropy-based statistics and diversity. |
| | `_entropy_discrete()`, `_entropy_continuous_histogram()` — Entropy helpers; `compute_entropy_stats()` — Real vs gen marginal entropy; `compute_prediction_entropy()` — Classifier prediction entropy; `compute_entropy_diversity()` — Diversity of gen set. |

---

**Synthetic_Data_Generation/** — *Brain signal generation (knowledge-based and model-based)*

**Knowledge_based/** — *Knowledge-based augmentation (no learned generator)*

| File | Description |
|------|-------------|
| `Baselines.py` | Simple augmentations and HHT/EMD utilities. |
| | `data_aug()` — Dispatch to noise/scale/flip/freq augmentations; `data_noise_f()`, `data_mult_f()`, `data_neg_f()`, `freq_mod_f()`, `freq_shift()` — Single-augmentation helpers; `HHTAnalysis()` — EMD + Hilbert visualization; `HHTFilter()` — EMD decomposition for HHT. |
| `DWTAug.py` | DWT-based cross-subject augmentation (script-style, db4 wavelet). |
| `DWTAug-ML.py` | Multi-level DWT augmentation (4-level decomposition). |
| `HHTAug.py` | HHT/EMD-based cross-subject augmentation (channel-level IMF reassembly). |
| `CR.py` | Cross-subject alignment and ML classification pipeline. |
| | `data_process()` — Load and align dataset; `data_alignment()`, `data_alignment_returnref()`, `data_alignment_passref()` — Alignment; `traintest_split_within_subject()` — Splits; `ml_classifier()` — Train/test classifier; `ml_within()` — Within-subject ML pipeline. |
| `utils/data_utils.py` | `traintest_split_cross_subject()`, `data_alignment()` — Cross-subject splits and alignment. |
| `utils/aug_utils.py` | `random_upsampling_transform()`, `leftrightflipping_transform()`, `leftrightdecay_transform()`, `small_laplace_normalize()` — Graph/geometry augmentations. |
| `utils/alg_utils.py` | `EA()`, `EA_online()`, `EA_ref()` — Euclidean alignment. |
| `Scale.py` | Scale/magnitude augmentation (multiply by constant). |
| | `data_mult_f()` — Scale samples by (1 − mult_mod) to generate augmented trials. |
| `Noise.py` | Additive noise augmentation. |
| | `data_noise_f()` — Add zero-mean random noise scaled by per-trial std to generate augmented trials. |
| `Flip.py` | Amplitude flip (sign flip + shift) augmentation. |
| | `data_neg_f()` — Negate amplitude and shift to non-negative for augmented trials. |
| `FShift.py` | Frequency shift augmentation via Hilbert transform. |
| | `nextpow2()` — Next power of two for FFT padding; `freq_shift()` — Shift signal in frequency (single segment); `freq_mod_f()` — Apply positive/negative frequency shift to dataset. |
| `FSurr.py` | Fourier surrogate (phase randomization) augmentation. |
| | `freqSur()` — FT surrogate using braindecode `FTSurrogate` for phase perturbation. |
| `HS.py` | Hemisphere / left–right channel recombination (dataset-specific). |
| | `hs_transform(dataset, X, y)` — Recombine left/middle/right channel groups per dataset (e.g. BNCI2014001, MI1-7, Weibo2014) to produce augmented samples and labels. |
| `DWT_reverse.py` | DWT-based augmentation variant that triples the training set. |
| | `DWTAug_reverse(Xs, Xt, ys, yt)` — DWT cross-subject reassembly (db4); returns [original Xs, augmented Xs, augmented Xt] and corresponding labels (effective 3× data). |

**Model_based/SSVEP_AUG/** — *Model-based augmentation (Vanilla/GAN/VAE generators with CNN/CNN-Transofrmer/CNN-LSTM structures + downstream evaluation)*

| Path | Description |
|------|-------------|
| **etc/** | *Configuration* |
| `config.yaml` | Training hyperparameters, model params, data paths, SSVEP/DL method settings. |
| `global_config.py` | `_init()` — Parse CLI and load YAML; `get_global_conf()` — Return config dict. |
| **Utils/** | *Data, training, and evaluation utilities* |
| `dataprocess.py` | Data loading, augmentation dispatch, and preprocessing. |
| | `fix_random_seed()` — Reproducibility; `data_aug()` — Dispatch (Noise/Scale/Flip/FShift/DWT/HHT/model-based); `data_noise_f()`, `data_mult_f()`, `data_neg_f()`, `freq_mod_f()`, `freq_shift()` — Simple aug; `DWTaug_multi()`, `HHTaug_multi()`, `DWTAug()`, `HHTAug()` — DWT/HHT multi-class; `HHTAnalysis()`, `HHTFilter()` — EMD; `dct_transform()`, `hs_transform()`, `freqSur()` — Other transforms; `data_preprocess()` — Full train/test preprocessing. |
| `EEGDataset.py` | SSVEP dataset and splits. |
| | `getSSVEPIntra` — Intra-subject dataset; `getSSVEPInter` — Inter-subject; `intra_benchmark40_split()`, `intra_benchmark12_split()` — Block splits; `cross_subject_split()` — Cross-subject folds; `z_score_normalization()` — Normalization. |
| `test.py` | Downstream evaluation (FBCCA, TRCA, DL models). |
| | `test()` — Main evaluation entry; `prepare_inputs()` — Tensor/numpy prep; `fbcca_evaluate()`, `trca_evaluate()` — FBCCA/TRCA accuracy; `model_evaluate()` — DL model eval; `train_on_batch()`, `train_on_batch_AUG()` — Train DL on raw/aug; `build_model()` — Instantiate DL model. |
| `Trainer.py` | Generator training pipeline. |
| | `build_generator()` — Create generator by arch (CNN / CNN-LSTM / CNN-Transformer); `train_aug_generator()` — Train GAN/VAE and save checkpoint. |
| `saveresult.py` | `train_params()` — Append param rows; `write_to_excel()` — Write accuracy table to Excel. |
| `Constraint.py` | `Conv2dWithConstraint`, `Spectral_Normalization`, `initialize_weights()` — Weight constraints for training. |
| `LossFunction.py` | `CELoss_Marginal_Smooth` — Classification loss with margin. |
| `Normalization.py` | Graph Laplacian and adjacency normalizations: `normalized_laplacian()`, `laplacian()`, `gcn()`, `aug_normalized_adjacency()`, `fetch_normalization()`, `row_normalize()`, etc. |
| `Script.py` | EEG/graph preprocessing and template utilities. |
| | `cal_Parameters()`, `norm_Data()`, `filter_Data()` — Utils; `get_Template_Signal()`, `EEG_Data_Segment()` — Templates and segments; `normalize_A()`, `generate_cheby_adj()`, `randomedge_drop()`, `preprocess_adj()` — Graph preprocessing. |
| `contrastive_loss.py` | Contrastive and metric losses: `NTXentLoss`, `NTXentLossNeg`, `NTXentLossClass`, `NTXentLossN`, `NTXentLoss_poly`, `hierarchical_contrastive_loss`, `ContrastiveLoss`, `TripletLoss`, `SupConLoss`, etc.; `guassian_kernel()`, `lmmd()` — Kernel/LMMD. |
| **Models/** | *Generator, discriminator, VAE, and downstream classifiers* |
| `Generator.py` | Generator architectures for denoising/augmentation. |
| | `EEGDenoiseGenerator_NoSeq` — CNN-only; `EEGDenoiseGenerator` — CNN + Bi-LSTM; `PositionalEncoding`, `EEGTransformer` — Transformer block; `EEGDenoiseGeneratorv2` — CNN + Transformer. |
| `Discriminator.py` | Discriminator and encoder backbones. |
| | `Spectral_Normalization`, `Conv2dWithConstraint` — Layers; `LSTM`, `ESNet` — LSTM/CNN; `PositionalEncoding`, `EEGTransformer` — Transformer; `ESNetv2` — Transformer-based discriminator. |
| `VAE.py` | VAE and variants. |
| | `ResidualLinearBlock` — Residual block; `VAE` — Encoder-decoder VAE; `VAEv2` — VAE with Transformer; `VAEShallowConvNet`, `VAE_IFNet` — VAE with ShallowConv/IFNet. |
| `TRANSfomer.py` | `DimensionExpandingTransformer` — Expands (B, T, 8Nc) to (B, T, 16Nc). |
| `IFNet.py`, `IFNetV2.py` | `IFNet` — Inter-frequency network (Conv, InterFre, Stem). |
| `ShallowConvNet.py` | `ShallowConvNet` — Shallow conv net for EEG. |
| **Models/KNoW/** | *Knowledge-based classifiers (downstream evaluation)* |
| `CCA.py` | `CCA_Base` — Canonical Correlation Analysis. |
| `TRCA.py` | `TRCA` — Task-related component analysis (`.trca()`, `.test_trca()`). |
| `TDCA.py` | `TDCA` — Task-discriminant component analysis; helpers: `isPD()`, `nearestPD()`, `robust_pattern()`, `xiang_dsp_kernel()`, `tdca_feature()`, `lagging_aug()`. |
| `FBCCA.py` | `FBCCA` — Filter-bank CCA (`.get_template_signal_with_labels()`, `.filter_bank()`, `.fbcca_classify()`). |
| `MSI.py` | `MSI_Base` — Multivariate synchronization index. |
| **Models/DeepL/** | *Deep learning classifiers (downstream evaluation)* |
| `EEGNet.py` | `EEGNet` — EEGNet (temporal + spatial conv, separable conv). |
| `CCNN.py` | `complex_spectrum_features()` — FFT features; `CNN` — Complex CNN. |
| `SSVEPNet.py` | `LSTM`, `ESNet` — SSVEP classification. |
| `SSVEPformer.py` | `PreNorm`, `FeedForward`, `Attention`, `Transformer`, `SSVEPformer`, `FFTLayer` — Transformer-based SSVEP. |
| `ConvCA.py` | `convca` — Convolutional CCA. |
| `FBtCNN.py` | `SamePadConv2d`, `tCNN` — Filter-bank tCNN. |
| `DDGCNN.py` | `GraphConvolution`, `DenseDDGCNN`, `DenseGCNNblock`, `GDCD`, `DCDGCN`, etc. — Graph CNN. |
| **Models/HZTKD/** | *Filter-bank and knowledge layers (bandpass, TRCA/TDCA in network)* |
| `Hztlayer.py` | `KLGlayer` — Knowledge layer; `BandpassConvLayer` — Bandpass conv; `FBlayer`, `MIXlayer` — Filter-bank weighting; `TRCAlayer`, `TDCAlayer` — TRCA/TDCA as layers. |

**Entry point:** `main.py` — `run()` — Orchestrates data load, augmentation, training, and evaluation.

---

## Installation

```bash
git clone https://github.com/wzwvv/DG4BCI.git
cd DG4BCI
pip install -r requirements.txt
```
