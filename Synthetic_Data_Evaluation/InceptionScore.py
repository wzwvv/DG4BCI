'''
=================================================
coding:utf-8
@File:      InceptionScore.py
@Author:    Ziwei Wang
@Function:  生成数据评测指标 - Inception Score (IS)
            IS = exp(E_x[ KL(p(y|x) || p(y)) ])，衡量生成样本的多样性与判别性
            对 EEG 使用特征空间上的分类器得到 p(y|x)
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
    计算 Inception Score（基于特征 + 分类器近似）。
    在真实数据上训练一个分类器，用其预测生成样本的 p(y|x)，再计算 IS。

    Parameters
    ----------
    real_features : np.ndarray, shape (N, D)
        真实样本的特征向量
    gen_features : np.ndarray, shape (M, D)
        生成样本的特征向量
    real_labels : np.ndarray, shape (N,), dtype int
        真实样本的类别标签，取值 [0, num_classes-1]
    num_classes : int
        类别数
    n_splits : int
        将生成样本分成 n_splits 份取平均，降低方差（默认 10）
    random_state : int
        随机种子

    Returns
    -------
    mean_is : float
        Inception Score 的均值（n_splits 次）
    std_is : float
        Inception Score 的标准差
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("InceptionScore 需要 sklearn: pip install scikit-learn")

    rng = np.random.default_rng(random_state)
    N, D = real_features.shape
    M = gen_features.shape[0]
    if len(real_labels) != N:
        raise ValueError(f"real_labels 长度 {len(real_labels)} 与 real_features 样本数 {N} 不一致")

    # 标准化特征（用真实集统计量）
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_features)
    gen_scaled = scaler.transform(gen_features)

    # 在真实数据上训练分类器，得到 p(y|x) 在生成数据上的估计
    clf = LogisticRegression(max_iter=500, random_state=random_state)
    clf.fit(real_scaled, real_labels)
    p_y_given_x = clf.predict_proba(gen_scaled)  # (M, num_classes)
    p_y_given_x = np.clip(p_y_given_x, 1e-8, 1.0)

    # 边缘分布 p(y)
    p_y = np.mean(p_y_given_x, axis=0)
    p_y = np.clip(p_y, 1e-8, 1.0)

    # 每个样本的 KL(p(y|x) || p(y))，再求 exp(mean(KL))
    kl = np.sum(p_y_given_x * (np.log(p_y_given_x) - np.log(p_y)), axis=1)
    if n_splits <= 1:
        mean_kl = np.mean(kl)
        return float(np.exp(mean_kl)), 0.0

    # 多份取平均
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
    简化版：无分类器，用特征离散化后的类别分布近似 p(y|x)，
    仅当无法提供 real_labels 时使用，结果与标准 IS 不可直接对比。

    Parameters
    ----------
    gen_features : np.ndarray, shape (M, D)
        生成样本的特征
    real_labels : optional
        未使用，仅为接口一致
    num_classes : optional
        未使用
    n_bins : int
        每维特征分箱数，得到 n_bins^D 个“伪类”（D 大时易爆炸，建议 D 较小或仅用部分维）

    Returns
    -------
    float
        基于直方图的多样性得分（越大越多样）
    """
    M, D = gen_features.shape
    if D > 5:
        # 仅用前几维避免维数灾难
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
    # quick test, 先提取特征再计算
    N, M, D = 100, 80, 24
    real_features = np.random.randn(N, D).astype(np.float32)  # replace with your festures
    gen_features = np.random.randn(M, D).astype(np.float32)  # replace with your festures
    real_labels = np.random.randint(0, 2, size=N)  # replace with your labels

    mean_is, std_is = compute_inception_score(
        real_features, gen_features, real_labels, num_classes=2, n_splits=5
    )
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")
