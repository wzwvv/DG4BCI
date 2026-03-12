'''
=================================================
coding:utf-8
@File:      SignalToNoiseRatio.py
@Author:    Ziwei Wang
@Function:  Evaluation metric for generated data - Signal-to-Noise Ratio (SNR).
            Measures the ratio of "signal strength" to "noise/variance" in feature or signal space.
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
    Define SNR in feature space.
    - mean_std: treat real as "signal", gen vs real difference as "noise":
      signal = ||mean(real)||, noise = std(gen - real_align), where real_align is pairing or mean alignment of real to gen.
    - Simplified: signal = mean(||real||_2) or std(real), noise = mean(||gen - mean(real)||), etc.

    Here we use: signal = mean of per-dimension std of real features (signal strength),
                 noise = norm of (gen - mean(real)) (how much generated samples deviate from real center),
                 SNR = signal / (noise + eps).

    Parameters
    ----------
    real_features : (N, D)
    gen_features : (M, D)
    method : "mean_std" | "power_ratio"
        mean_std: signal = mean of per-dim std of real, noise = mean distance from gen to real mean
        power_ratio: use variance ratio

    Returns
    -------
    float
        SNR (scalar); higher means generated data is closer to real distribution center/scale.
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
        # Mean distance of generated samples to real mean (as "noise")
        diff = gen - mu_r
        noise = np.sqrt(np.mean(diff ** 2)) + eps
        return float(signal / noise)

    # power_ratio: real variance / MSE of gen relative to real mean
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
    Compute SNR in raw EEG signal space.
    Treat real as reference signal; difference (gen - real) as noise (e.g. in paired conditional generation).
    If not paired: use real variance as signal power, gen variance or (gen - mean(real)) as noise.

    Parameters
    ----------
    real_eeg : np.ndarray, shape (N, C, T) or (N, T)
        Real EEG.
    gen_eeg : np.ndarray, shape (M, C, T) or (M, T)
        Generated EEG; if N==M can be treated as one-to-one paired.
    per_channel : bool
        If True, return per-channel SNR (keep channel axis for (N,C,T)).
    axis_time : int
        Time axis index for computing power over time.

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
        # One-to-one: noise = gen - real
        signal_power = np.mean(real ** 2, axis=axis_time)
        noise = gen - real
        noise_power = np.mean(noise ** 2, axis=axis_time) + eps
    else:
        # Unpaired: signal = real variance, noise = gen deviation from real mean
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
    Simplest form: signal = mean variance of real features, noise = mean variance of (gen - mean(real)),
    SNR = signal / noise, then convert to dB.
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
