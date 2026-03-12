# Designer:Yudong Pan
# Coder:God's hand
# Time:2024/2/3 23:45
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as  plt
import einops
from scipy import signal
import math
import argparse
import sys

class PreNorm(nn.Module):
    def __init__(self, token_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, token_length, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_length, token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, token_num, token_length, kernal_length, dropout):
        super().__init__()

        self.att2conv = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.att2conv(x)
        return out


class Transformer(nn.Module):
    def __init__(self, depth, token_num, token_length, kernal_length, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(token_length, Attention(token_num, token_length, kernal_length, dropout=dropout)),
                PreNorm(token_length, FeedForward(token_length, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SSVEPformer(nn.Module):
    def __init__(self, depth, attention_kernal_length, chs_num, class_num,FFT_PARAMS, dropout,resolution=0.2, start_freq=8, end_freq=64):
        super().__init__()
        token_num = chs_num * 2
        token_dim = 2*round((end_freq-start_freq)/resolution)
        self.complex_spectrum_features=FFTLayer(FFT_PARAMS=FFT_PARAMS,resolution=resolution, start_freq=start_freq, end_freq=end_freq)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(chs_num, token_num, 1, padding=1 // 2, groups=1),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.transformer = Transformer(depth, token_num, token_dim, attention_kernal_length, dropout)

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * token_num, class_num * 6),
            nn.LayerNorm(class_num * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num * 6, class_num)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x= self.complex_spectrum_features(x)
        x = self.to_patch_embedding(x)
        x = self.transformer(x)
        return self.mlp_head(x)

class FFTLayer(nn.Module):
    def __init__(self,FFT_PARAMS, resolution=0.2, start_freq=8, end_freq=64):
        super(FFTLayer, self).__init__()
        self.FFT_PARAMS = FFT_PARAMS
        self.resolution, self.start_freq, self.end_freq = resolution, start_freq, end_freq

    def forward(self, segmented_data):
        sample_freq = self.FFT_PARAMS[0]
        time_len = self.FFT_PARAMS[1]
        resolution, start_freq, end_freq =  self.resolution, self.start_freq, self.end_freq
        NFFT = round(sample_freq / resolution)
        fft_index_start = int(round(start_freq / resolution))
        fft_index_end = int(round(end_freq / resolution)) + 1
        sample_point = int(sample_freq * time_len)
        fft_result = torch.fft.fft(segmented_data[:,0,:,:], n=NFFT) / (sample_point / 2)
        real_part = torch.real(fft_result[:, :, fft_index_start:fft_index_end - 1])
        imag_part = torch.imag(fft_result[:, :, fft_index_start:fft_index_end - 1])
        features_data = torch.cat([real_part, imag_part], dim=-1)  #
        fft_signal = torch.abs(fft_result[:,:,40:160])
        # print(fft_signal.shape)
        # for _ in range(30):
        # # Plot spectrum
        #     plt.figure(figsize=(20, 8))
        #     plt.plot([i for i in range(fft_signal.shape[-1])], fft_signal[_].cpu().T, marker='o', label="FFT Magnitude Spectrum")
        #     plt.xlabel("Frequency (Hz)")
        #     plt.ylabel("Magnitude")
        #     plt.title("FFT Spectrum (6-16 Hz)")
        #     plt.xticks(np.arange(0, 120, 1))  # set ticks
        #     plt.grid()
        #     plt.legend()
        #     plt.show()


        return features_data