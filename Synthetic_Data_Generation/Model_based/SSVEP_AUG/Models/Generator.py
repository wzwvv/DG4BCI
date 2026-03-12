import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGDenoiseGenerator_NoSeq(nn.Module):
    """
        CNN Baseline
    """

    def __init__(self, Nc, dropout=0.5):
        super().__init__()
        self.Nc = Nc

        # ---------- Encoder ----------
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 2*Nc, kernel_size=(Nc, 1), stride=(Nc, 1)),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.temp_conv1 = nn.Sequential(
            nn.Conv2d(2*Nc, 4*Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(4*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.temp_conv2 = nn.Sequential(
            nn.Conv2d(4*Nc, 8*Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(8*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # ---------- Bottleneck (pure CNN) ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(8*Nc, 16*Nc, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16*Nc),
            nn.PReLU()
        )

        # ---------- Decoder ----------
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                16*Nc, 4*Nc,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=(0, 1)
            ),
            nn.BatchNorm2d(4*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                8*Nc, 2*Nc,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=(0, 1)
            ),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(4*Nc, 2*Nc, kernel_size=(Nc, 1), stride=(Nc, 1)),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.final_conv = nn.Conv2d(2*Nc, 1, kernel_size=(1, 1))

    @staticmethod
    def _align_time(a, b):
        """Crop to same time length."""
        Ta, Tb = a.shape[-1], b.shape[-1]
        T = min(Ta, Tb)
        return a[..., :T], b[..., :T]

    def forward(self, x):
        # x: (B, 1, Nc, T)

        # -------- Encoder --------
        x1 = self.spatial_conv(x)     # (B, 2Nc, 1, T)
        x2 = self.temp_conv1(x1)      # (B, 4Nc, 1, T/2)
        x3 = self.temp_conv2(x2)      # (B, 8Nc, 1, T/4)

        # -------- Bottleneck --------
        b = self.bottleneck(x3)       # (B, 16Nc, 1, T/4)

        # -------- Statistics (keep original interface) --------
        feat = b.squeeze(2)           # (B, C, T')
        mean1 = feat.mean(dim=2, keepdim=True)
        variance1 = feat.var(dim=2, keepdim=True, unbiased=True)
        mean2 = feat.mean(dim=1, keepdim=True)
        variance2 = feat.var(dim=1, keepdim=True, unbiased=True)

        # -------- Decoder --------
        u1 = self.deconv1(b)
        u1, x2a = self._align_time(u1, x2)
        e1 = torch.cat([u1, x2a], dim=1)

        u2 = self.deconv2(e1)
        u2, x1a = self._align_time(u2, x1)
        e2 = torch.cat([u2, x1a], dim=1)

        u3 = self.spatial_deconv(e2)
        output = self.final_conv(u3)

        return mean1, variance1, mean2, variance2, output

class EEGDenoiseGenerator(nn.Module):
    def __init__(self, Nc,Nt, dropout=0.5):
        super(EEGDenoiseGenerator, self).__init__()
        self.Nc = Nc
        self.dropout = dropout

        # ----------- Encoder ----------- #
        # Block 1: Spatial Conv
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 2*Nc, kernel_size=(Nc,1), stride=(Nc,1)),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 2: Temporal Conv (1)
        self.temp_conv1 = nn.Sequential(
            nn.Conv2d(2*Nc, 4*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7)),
            nn.BatchNorm2d(4*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 3: Temporal Conv (2)
        self.temp_conv2 = nn.Sequential(
            nn.Conv2d(4*Nc, 8*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7)),
            nn.BatchNorm2d(8*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 4: Bi-LSTM
        self.bi_lstm = nn.LSTM(
            input_size=8*Nc,
            hidden_size=8*Nc,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.avgpool = nn.AvgPool1d(kernel_size=2)

        # ----------- Decoder ----------- #
        # Block 5: Temporal Transpose Conv
        if Nt % 4 == 0:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(16*Nc, 4*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7), output_padding=(0,0)),
                nn.BatchNorm2d(4*Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(16*Nc, 4*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7), output_padding=(0,1)),
                nn.BatchNorm2d(4*Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )

        # Block 6: Temporal Transpose Conv
        if Nt % 2 == 0:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(8*Nc, 2*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7), output_padding=(0,0)),
                nn.BatchNorm2d(2*Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(8*Nc, 2*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7), output_padding=(0,1)),
                nn.BatchNorm2d(2*Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )

        # Block 7: Spatial Transpose Conv
        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(4*Nc, 2*Nc, kernel_size=(Nc,1), stride=(Nc,1)),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 8: Final Temporal Conv
        self.final_conv = nn.Conv2d(2*Nc, 1, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        # x shape: (B, 1, Nc, T)

        # ----------- Encoding Path ----------- #
        x1 = self.spatial_conv(x)             # -> (B, 2Nc, 1, T)
        x2 = self.temp_conv1(x1)              # -> (B, 4Nc, 1, T/2)
        x3 = self.temp_conv2(x2)              # -> (B, 8Nc, 1, T/4)

        # Prepare for Bi-LSTM
        x3_squeezed = x3.squeeze(2).permute(0, 2, 1)  # -> (B, T/4, 8Nc)
        lstm_out, _ = self.bi_lstm(x3_squeezed)       # -> (B, T/4, 16Nc)
        lstm_out = self.avgpool(lstm_out)
        mean1 = torch.mean(lstm_out.squeeze(2), dim=1, keepdim=True)  # shape (B, C, 1)

        # Variance (unbiased)
        variance1 = torch.var(lstm_out.squeeze(2), dim=1, keepdim=True, unbiased=True)  # shape (B, C, 1)
        
        mean2 = torch.mean(lstm_out.squeeze(2), dim=-1, keepdim=True)  # shape (B, 1)

        # Variance (unbiased)
        variance2 = torch.var(lstm_out.squeeze(2), dim=-1, keepdim=True, unbiased=True)  # shape (B, C, 1)
        lstm_out = lstm_out.permute(0, 2, 1).unsqueeze(2)  # -> (B, 16Nc, 1, T/4)

        e0 = torch.cat([lstm_out, x3], dim=1)         # -> (B, 24Nc, 1, T/4)

        # ----------- Decoding Path ----------- #
        u1 = self.deconv1(e0)                         # -> (B, 4Nc, 1, T/2)
        e1 = torch.cat([u1, x2], dim=1)               # -> (B, 8Nc, 1, T/2)
        

        u2 = self.deconv2(e1)                      # -> (B, 2Nc, 1, T)
        e2 = torch.cat([u2, x1], dim=1)               # -> (B, 4Nc, 1, T)
        u3 = self.spatial_deconv(e2)                  # -> (B, 2Nc, Nc, T)
        output = self.final_conv(u3)                  # -> (B, 1, Nc, T)

        return mean1,variance1,mean2,variance2,output
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, T, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, C)
        return x + self.pe[:, :x.size(1)]

class EEGTransformer(nn.Module):
    def __init__(self, Nc, dropout=0.5, num_layers=4):
        super().__init__()
        d_model = 8 * Nc

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (B, T, C)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        return x
class EEGDenoiseGeneratorv2(nn.Module):
    def __init__(self, Nc, Nt, dropout=0.5):
        super(EEGDenoiseGeneratorv2, self).__init__()
        self.Nc = Nc
        self.dropout = dropout

        # ----------- Encoder ----------- #
        # Block 1: Spatial Conv
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 2 * Nc, kernel_size=(Nc, 1), stride=(Nc, 1)),
            nn.BatchNorm2d(2 * Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 2: Temporal Conv (1)
        self.temp_conv1 = nn.Sequential(
            nn.Conv2d(2 * Nc, 4 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(4 * Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 3: Temporal Conv (2)
        self.temp_conv2 = nn.Sequential(
            nn.Conv2d(4 * Nc, 8 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(8 * Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # self.transformer_layer = TransformerEncoder(
        #     TransformerEncoderLayer(
        #         d_model=8 * Nc,
        #         nhead=12,
        #         dim_feedforward=1024,
        #         dropout=dropout,
        #         batch_first=True  # use (B, T, C) format
        #     ),
        #     num_layers=4
        # )
        self.transformer_layer = EEGTransformer(Nc=Nc,dropout=dropout)
        self.avgpool = nn.AvgPool1d(kernel_size=2)

        # ----------- Decoder ----------- #
        # Block 5: Temporal Transpose Conv
        if Nt % 4 == 0:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(16 * Nc, 4 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 0)),
                nn.BatchNorm2d(4 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(16 * Nc, 4 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 1)),
                nn.BatchNorm2d(4 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )

        # Block 6: Temporal Transpose Conv
        if Nt % 2 == 0:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(8 * Nc, 2 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 0)),
                nn.BatchNorm2d(2 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(8 * Nc, 2 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 1)),
                nn.BatchNorm2d(2 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )

        # Block 7: Spatial Transpose Conv
        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(4 * Nc, 2 * Nc, kernel_size=(Nc, 1), stride=(Nc, 1)),
            nn.BatchNorm2d(2 * Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 8: Final Temporal Conv
        self.final_conv = nn.Conv2d(2 * Nc, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        # x shape: (B, 1, Nc, T)

        # ----------- Encoding Path ----------- #
        x1 = self.spatial_conv(x)  # -> (B, 2Nc, 1, T)
        x2 = self.temp_conv1(x1)  # -> (B, 4Nc, 1, T/2)
        x3 = self.temp_conv2(x2)  # -> (B, 8Nc, 1, T/4)

        x3_squeezed = x3.squeeze(2).permute(0, 2, 1)  # -> (B, T/4, 8Nc)
        transformer_out= self.transformer_layer(x3_squeezed)  # -> (B, T/4, 16Nc)
        #lstm_out = self.avgpool(lstm_out)
        transformer_out = transformer_out.permute(0, 2, 1).unsqueeze(2)  # -> (B, 16Nc, 1, T/4)

        e0 = torch.cat([transformer_out, x3], dim=1)  # -> (B, 24Nc, 1, T/4)
        mean1 = torch.mean(transformer_out.squeeze(2), dim=1, keepdim=True)  # shape (B, C, 1)

        # Variance (unbiased)
        variance1 = torch.var(transformer_out.squeeze(2), dim=1, keepdim=True, unbiased=True)  # shape (B, C, 1)
        
        mean2 = torch.mean(transformer_out.squeeze(2), dim=-1, keepdim=True)  # shape (B, 1)

        # Variance (unbiased)
        variance2 = torch.var(transformer_out.squeeze(2), dim=-1, keepdim=True, unbiased=True)  # shape (B, C, 1)

        # Log variance (small constant to avoid log(0))
        log_variance1 = torch.log(variance1 + 1e-8)  # shape (B, C, 1)
        log_variance2 = torch.log(variance2 + 1e-8)  # shape (B, C, 1)
        # ----------- Decoding Path ----------- #
        u1 = self.deconv1(e0)  # -> (B, 4Nc, 1, T/2)
        e1 = torch.cat([u1, x2], dim=1)  # -> (B, 8Nc, 1, T/2)

        u2 = self.deconv2(e1)  # -> (B, 2Nc, 1, T)
        e2 = torch.cat([u2, x1], dim=1)  # -> (B, 4Nc, 1, T)
        u3 = self.spatial_deconv(e2)  # -> (B, 2Nc, Nc, T)
        output = self.final_conv(u3)  # -> (B, 1, Nc, T)

        return mean1,log_variance1,mean2,log_variance2,output
    