'''
=================================================
coding:utf-8
@File:      EntropyStats.py
@Author:    Ziwei Wang
@Function:  Evaluation metric for generated data - Entropy-based statistics.
            Higher entropy means more diversity; difference from real entropy measures distribution match.
=================================================
'''

import numpy as np
from typing import Tuple, Optional


def _entropy_discrete(probs: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Entropy of discrete distribution H = -sum(p*log(p)); probs sum to 1 along axis."""
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=axis)


def _entropy_continuous_histogram(x: np.ndarray, n_bins: int = 50, axis: int = -1) -> float:
    """Estimate entropy of 1D or multidimensional distribution via histogram (continuous approximation)."""
    x = np.asarray(x).reshape(-1)
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist + 1e-8
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log(hist)))


def compute_entropy_stats(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    n_bins: int = 50,
) -> Tuple[float, float, float]:
    """
    Entropy-based statistics in feature space.

    Parameters
    ----------
    real_features : np.ndarray, shape (N, D)
    gen_features : np.ndarray, shape (M, D)
    n_bins : int
        Number of bins for marginal entropy estimation.

    Returns
    -------
    real_entropy : float
        Mean marginal entropy of real feature distribution (per-dim histogram).
    gen_entropy : float
        Mean marginal entropy of generated feature distribution.
    entropy_diff : float
        |gen_entropy - real_entropy|; lower means closer match.
    """
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if real.ndim == 1:
        real = real.reshape(1, -1)
    if gen.ndim == 1:
        gen = gen.reshape(1, -1)
    n_r, d = real.shape
    n_g, d_g = gen.shape
    if d != d_g:
        raise ValueError(f"Feature dimension mismatch: real {d} vs gen {d_g}")

    real_entropies = []
    gen_entropies = []
    for j in range(d):
        real_entropies.append(_entropy_continuous_histogram(real[:, j], n_bins=n_bins))
        gen_entropies.append(_entropy_continuous_histogram(gen[:, j], n_bins=n_bins))
    real_entropy = float(np.mean(real_entropies))
    gen_entropy = float(np.mean(gen_entropies))
    entropy_diff = float(np.abs(gen_entropy - real_entropy))
    return real_entropy, gen_entropy, entropy_diff


def compute_prediction_entropy(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    real_labels: np.ndarray,
    num_classes: int,
) -> Tuple[float, float]:
    """
    Train classifier on real data, predict on generated data, compute entropy of prediction distribution.
    - High prediction entropy: generated samples hard to classify (diverse or noisy).
    - Can compare with prediction entropy on real set.

    Parameters
    ----------
    real_features : (N, D)
    gen_features : (M, D)
    real_labels : (N,) int
    num_classes : int

    Returns
    -------
    real_pred_entropy : float
        Mean prediction entropy on real samples.
    gen_pred_entropy : float
        Mean prediction entropy on generated samples.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("EntropyStats requires sklearn: pip install scikit-learn")

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_features)
    gen_scaled = scaler.transform(gen_features)
    clf = LogisticRegression(max_iter=500)
    clf.fit(real_scaled, real_labels)
    p_real = np.clip(clf.predict_proba(real_scaled), 1e-8, 1.0)
    p_gen = np.clip(clf.predict_proba(gen_scaled), 1e-8, 1.0)
    real_pred_entropy = float(np.mean(_entropy_discrete(p_real, axis=1)))
    gen_pred_entropy = float(np.mean(_entropy_discrete(p_gen, axis=1)))
    return real_pred_entropy, gen_pred_entropy


def compute_entropy_diversity(gen_features: np.ndarray, n_bins: int = 30) -> float:
    """
    For generated set only: compute (mean marginal) entropy of feature distribution as diversity metric.
    Higher means more spread of the generated distribution.

    Parameters
    ----------
    gen_features : (M, D)
    n_bins : int

    Returns
    -------
    float
    """
    gen = np.asarray(gen_features).reshape(-1, gen_features.shape[-1])
    entropies = [_entropy_continuous_histogram(gen[:, j], n_bins=n_bins) for j in range(gen.shape[1])]
    return float(np.mean(entropies))


if __name__ == "__main__":
    # quick test
    N, M, D = 100, 80, 24
    real_features = np.random.randn(N, D).astype(np.float32)
    gen_features = np.random.randn(M, D).astype(np.float32)
    real_labels = np.random.randint(0, 2, size=N)

    real_h, gen_h, diff = compute_entropy_stats(real_features, gen_features, n_bins=30)
    print(f"Real entropy: {real_h:.4f}, Gen entropy: {gen_h:.4f}, |diff|: {diff:.4f}")

    real_pe, gen_pe = compute_prediction_entropy(
        real_features, gen_features, real_labels, num_classes=2
    )
    print(f"Prediction entropy — real: {real_pe:.4f}, gen: {gen_pe:.4f}")
