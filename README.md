# Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions
This repository contains the source code for our paper **Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions**.

This survey provides a comprehensive review of brain signal generation for BCIs, covering methodological taxonomies, benchmark experiments, evaluation metrics, and key applications. We systematically categorize existing generative algorithms into four types: knowledge-based, feature-based, model-based, and translation-based approaches. Furthermore, we benchmark existing brain signal generation approaches across four representative BCI paradigms to provide an objective performance comparison. Finally, we discuss the potentials and challenges of current generation approaches and prospect future research on accurate, data-efficient, and privacy-aware BCI systems. 

## Data scarcity in BCI
Recent success of deep learning underscores the importance of large, high-quality training data. For example, Google’s trillion-word corpus has significantly boosted performance in language models. However, acquiring sufficient brain signals presents challenges.
<img width="592" height="622" alt="image" src="https://github.com/user-attachments/assets/018b5b9b-6fe0-45b5-8a9f-b29d6568ab5c" />

<p align="center"><font color="gray">Figure 1: Data scarcity issue in BCI.</font></p>

## Framework
Data generation driven machine learning pipeline for BCIs, which includes brain signal acquisition, data preprocessing, data generation, feature engineering, and classification/regression. The latter two components can be unified into a single end-to-end neural network. Data generation approaches are categorized into four types: (a) knowledge-based generation, (b) feature-based generation, (c) model-based generation, and (d) translation-based generation.

<img width="1075" height="639" alt="image" src="https://github.com/user-attachments/assets/b0793e9f-541d-45eb-a58b-61c841d4367f" />

<p align="center"><font color="gray">Figure 2: Data generation driven machine learning pipeline for BCIs.</font></p>

