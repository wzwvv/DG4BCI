'''
=================================================
coding:utf-8
@File:      CosineSimilarity.py
@Author:    Ziwei Wang
@Function:  Evaluation metric for generated data - Cosine similarity.
            Measures similarity between real and generated features; higher is more similar.
=================================================
'''

import numpy as np
from typing import Optional, Union


def compute_cosine_similarity(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    aggregate: str = "mean",
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Compute cosine similarity between real and generated features.

    Parameters
    ----------
    real_features : np.ndarray, shape (N, D)
        Real sample feature vectors.
    gen_features : np.ndarray, shape (M, D)
        Generated sample feature vectors.
    aggregate : str
        - "mean": compute cosine for (real_i, gen_j) or (real_mean, gen_mean) then aggregate
        - "mean_vectors": compare only the two set mean vectors, return scalar
        - "pairwise_mean": for each real, average cosine to all gen (or bidirectional)
    axis : optional
        If given, normalize along this axis before computing (default None).

    Returns
    -------
    float or np.ndarray
        Scalar or array.
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

    def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        return np.sum(an * bn, axis=-1)

    if aggregate == "mean_vectors":
        mu_r = np.mean(real, axis=0)
        mu_g = np.mean(gen, axis=0)
        return float(_cos(mu_r[None], mu_g[None])[0])

    if aggregate == "pairwise_mean":
        # Cosine of each real to all gen, then average
        cos_all = np.zeros(n_r)
        for i in range(n_r):
            c = _cos(gen, np.broadcast_to(real[i], (n_g, d)))
            cos_all[i] = np.mean(c)
        return float(np.mean(cos_all))

    # default: mean_vectors
    mu_r = np.mean(real, axis=0)
    mu_g = np.mean(gen, axis=0)
    return float(_cos(mu_r[None], mu_g[None])[0])


def compute_cosine_similarity_per_sample(
    real_features: np.ndarray,
    gen_features: np.ndarray,
) -> float:
    """
    Requires real and gen to have the same size and be one-to-one paired; compute mean cosine similarity per pair.
    Suitable for conditional generation where real[i] and gen[i] share the same condition.

    Parameters
    ----------
    real_features : (N, D)
    gen_features : (N, D)

    Returns
    -------
    float
        Mean cosine similarity in [-1, 1]; higher is more similar.
    """
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if real.shape != gen.shape:
        raise ValueError(f"Shapes must match: real {real.shape} vs gen {gen.shape}")
    n = real.shape[0]
    rn = real / (np.linalg.norm(real, axis=1, keepdims=True) + 1e-8)
    gn = gen / (np.linalg.norm(gen, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(rn * gn, axis=1)))


if __name__ == "__main__":
    # quick test
    N, D = 50, 24
    real_features = np.random.randn(N, D).astype(np.float32)
    gen_features = np.random.randn(N, D).astype(np.float32)

    cs_mean = compute_cosine_similarity(real_features, gen_features, aggregate="mean_vectors")
    print(f"Cosine similarity (mean vectors): {cs_mean:.4f}")

    cs_pair = compute_cosine_similarity_per_sample(real_features, gen_features)
    print(f"Cosine similarity (per-sample, 1-to-1): {cs_pair:.4f}")
