'''
=================================================
coding:utf-8
@File:      SignalToNoiseRatio.py
@Author:    Ziwei Wang
@Function:  生成数据评测指标 - 信噪比 (SNR)
            在特征空间或信号空间衡量“信号强度”相对“噪声/变异”的比值
=================================================
'''

import numpy as np
from typing import Optional, Literal, Union


def compute_snr_features(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    method: Literal["mean_std", "power_ratio"] = "mean_std",
) -> float:
    """
    在特征空间上定义 SNR。
    - mean_std: 将 real 视为“信号”，gen 与 real 的差异视为“噪声”：
      signal = ||mean(real)||, noise = std(gen - real_align)，其中 real_align 为 real 对 gen 的配对或均值对齐
    - 简化：signal = mean(||real||_2) 或 std(real)，noise = mean(||gen - mean(real)||) 等。

    这里采用：signal = 真实特征各维的标准差均值（信号强度），
             noise = 生成与真实均值之差的范数（生成偏离真实中心的程度），
             SNR = signal / (noise + eps)。

    Parameters
    ----------
    real_features : (N, D)
    gen_features : (M, D)
    method : "mean_std" | "power_ratio"
        mean_std: signal=real 各维 std 的均值，noise=gen 到 real 均值的距离的均值
        power_ratio: 用方差比

    Returns
    -------
    float
        SNR（标量），越大表示生成越接近真实分布的中心/尺度
    """
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if real.ndim == 1:
        real = real.reshape(1, -1)
    if gen.ndim == 1:
        gen = gen.reshape(1, -1)
    eps = 1e-8
    mu_r = np.mean(real, axis=0)

    if method == "mean_std":
        signal = np.mean(np.std(real, axis=0)) + eps
        # 生成样本到真实均值的平均距离（作为“噪声”）
        diff = gen - mu_r
        noise = np.sqrt(np.mean(diff ** 2)) + eps
        return float(signal / noise)

    # power_ratio: 真实方差 / 生成相对真实均值的均方误差
    var_r = np.mean(np.var(real, axis=0)) + eps
    mse_gen = np.mean(np.sum((gen - mu_r) ** 2, axis=1)) + eps
    return float(var_r / mse_gen)


def compute_snr_eeg(
    real_eeg: np.ndarray,
    gen_eeg: np.ndarray,
    per_channel: bool = False,
    axis_time: int = -1,
) -> Union[float, np.ndarray]:
    """
    在原始 EEG 信号空间计算 SNR。
    将 real 视为参考信号，gen 与 real 的差异视为噪声（例如条件生成中一一对应时）。
    若不对应，则用 real 的方差作为信号功率，gen 的方差或 gen 与 real 均值的差作为噪声。

    Parameters
    ----------
    real_eeg : np.ndarray, shape (N, C, T) 或 (N, T)
        真实 EEG
    gen_eeg : np.ndarray, shape (M, C, T) 或 (M, T)
        生成 EEG，若 N==M 可视为一一对应
    per_channel : bool
        True 时返回每通道的 SNR（对 (N,C,T) 在 C 上保留）
    axis_time : int
        时间轴索引，用于在时间上求功率

    Returns
    -------
    float or np.ndarray
        SNR(dB) = 10 * log10(signal_power / noise_power + eps)
    """
    real = np.asarray(real_eeg, dtype=np.float64)
    gen = np.asarray(gen_eeg, dtype=np.float64)
    eps = 1e-12
    n_r, n_g = real.shape[0], gen.shape[0]

    if real.shape == gen.shape and n_r == n_g:
        # 一一对应：噪声 = gen - real
        signal_power = np.mean(real ** 2, axis=axis_time)
        noise = gen - real
        noise_power = np.mean(noise ** 2, axis=axis_time) + eps
    else:
        # 不对应：信号 = real 的方差，噪声 = gen 相对 real 均值的偏差
        mu_r = np.mean(real, axis=0)
        signal_power = np.mean((real - mu_r) ** 2, axis=0) + eps
        diff = gen - mu_r
        noise_power = np.mean(diff ** 2, axis=0) + eps

    snr_linear = signal_power / noise_power
    snr_db = 10.0 * np.log10(snr_linear + eps)

    if per_channel and snr_db.ndim > 0:
        return snr_db
    return float(np.mean(snr_db))


def compute_snr_simple(real_features: np.ndarray, gen_features: np.ndarray, eps: float = 1e-8) -> float:
    """
    最简单形式：signal = real 特征的方差均值，noise = (gen - mean(real)) 的方差均值，
    SNR = signal / noise，再转为 dB。
    """
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    mu_r = np.mean(real, axis=0)
    var_signal = np.mean(np.var(real, axis=0)) + eps
    var_noise = np.mean(np.var(gen - mu_r, axis=0)) + eps
    return float(10.0 * np.log10(var_signal / var_noise + eps))


if __name__ == "__main__":
    # quick test
    N, D = 50, 24
    real_features = np.random.randn(N, D).astype(np.float32) # replace with your features
    gen_features = np.random.randn(N, D).astype(np.float32) # replace with your features
    gen_features = np.random.randn(N, D).astype(np.float32) * 0.5 + np.mean(real_features, axis=0)

    snr_f = compute_snr_features(real_features, gen_features, method="mean_std")
    print(f"SNR (features, mean_std): {snr_f:.4f}")

    snr_db = compute_snr_simple(real_features, gen_features)
    print(f"SNR (simple, dB): {snr_db:.4f}")

    # EEG (N, C, T)
    N, C, T = 20, 8, 256
    real_eeg = np.random.randn(N, C, T).astype(np.float32)
    gen_eeg = real_eeg + np.random.randn(N, C, T).astype(np.float32) * 0.3
    snr_eeg = compute_snr_eeg(real_eeg, gen_eeg, per_channel=False)
    print(f"SNR (EEG 1-to-1, dB): {snr_eeg:.4f}")
