'''
=================================================
coding:utf-8
@File:      InceptionScore.py
@Author:    Ziwei Wang
@Function:  Evaluation metric for generated data - Inception Score (IS).
            IS = exp(E_x[ KL(p(y|x) || p(y)) ]), measures diversity and discriminability of generated samples.
            For EEG, p(y|x) is obtained from a classifier in feature space.
=================================================
'''

import numpy as np
from typing import Optional, Tuple


def compute_inception_score(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    real_labels: np.ndarray,
    num_classes: int,
    n_splits: int = 10,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Compute Inception Score (feature + classifier approximation).
    Train a classifier on real data, predict p(y|x) on generated samples, then compute IS.

    Parameters
    ----------
    real_features : np.ndarray, shape (N, D)
        Real sample feature vectors.
    gen_features : np.ndarray, shape (M, D)
        Generated sample feature vectors.
    real_labels : np.ndarray, shape (N,), dtype int
        Real sample class labels in [0, num_classes-1].
    num_classes : int
        Number of classes.
    n_splits : int
        Split generated samples into n_splits for averaging to reduce variance (default 10).
    random_state : int
        Random seed.

    Returns
    -------
    mean_is : float
        Mean Inception Score over n_splits.
    std_is : float
        Std of Inception Score.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("InceptionScore requires sklearn: pip install scikit-learn")

    rng = np.random.default_rng(random_state)
    N, D = real_features.shape
    M = gen_features.shape[0]
    if len(real_labels) != N:
        raise ValueError(f"real_labels length {len(real_labels)} does not match real_features sample count {N}")

    # Standardize features using real-set statistics
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_features)
    gen_scaled = scaler.transform(gen_features)

    # Train classifier on real data to get p(y|x) on generated data
    clf = LogisticRegression(max_iter=500, random_state=random_state)
    clf.fit(real_scaled, real_labels)
    p_y_given_x = clf.predict_proba(gen_scaled)  # (M, num_classes)
    p_y_given_x = np.clip(p_y_given_x, 1e-8, 1.0)

    # Marginal p(y)
    p_y = np.mean(p_y_given_x, axis=0)
    p_y = np.clip(p_y, 1e-8, 1.0)

    # Per-sample KL(p(y|x) || p(y)), then exp(mean(KL))
    kl = np.sum(p_y_given_x * (np.log(p_y_given_x) - np.log(p_y)), axis=1)
    if n_splits <= 1:
        mean_kl = np.mean(kl)
        return float(np.exp(mean_kl)), 0.0

    # Average over splits
    scores = []
    indices = rng.permutation(M)
    split_size = M // n_splits
    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < n_splits - 1 else M
        if start >= end:
            continue
        sub_kl = kl[indices[start:end]]
        scores.append(np.exp(np.mean(sub_kl)))
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.std(scores))


def compute_inception_score_simple(
    gen_features: np.ndarray,
    real_labels: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    n_bins: int = 10,
) -> float:
    """
    Simplified: no classifier; approximate p(y|x) by discretized feature histogram.
    Use when real_labels are not available; results are not directly comparable to standard IS.

    Parameters
    ----------
    gen_features : np.ndarray, shape (M, D)
        Generated sample features.
    real_labels : optional
        Unused; for API consistency.
    num_classes : optional
        Unused.
    n_bins : int
        Bins per dimension; yields n_bins^D "pseudo-classes" (use small D or subset of dims to avoid explosion).

    Returns
    -------
    float
        Histogram-based diversity score (higher = more diverse).
    """
    M, D = gen_features.shape
    if D > 5:
        # Use first few dimensions to avoid curse of dimensionality
        gen_features = gen_features[:, :5].copy()
        D = 5
    bins_per_dim = min(n_bins, max(2, M // 10))
    hist, _ = np.histogramdd(gen_features, bins=bins_per_dim)
    hist = hist.ravel()
    hist = hist + 1e-8
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log(hist))
    return float(np.exp(entropy))


if __name__ == "__main__":
    # Quick test: extract features first then compute
    N, M, D = 100, 80, 24
    real_features = np.random.randn(N, D).astype(np.float32)  # replace with your festures
    gen_features = np.random.randn(M, D).astype(np.float32)  # replace with your festures
    real_labels = np.random.randint(0, 2, size=N)  # replace with your labels

    mean_is, std_is = compute_inception_score(
        real_features, gen_features, real_labels, num_classes=2, n_splits=5
    )
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")
