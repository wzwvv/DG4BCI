import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ResidualLinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)

        # 用于调整残差连接维度的投影层（如果需要）
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.linear1(x)
        out = F.leaky_relu(out, 0.2)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.layer_norm(out + identity)  # 残差连接 + 层归一化
        out = F.leaky_relu(out, 0.2)
        out = self.dropout(out)

        return out
class VAE(nn.Module):
    def __init__(self, Nc, Nt, latent_dim=24, dropout=0.5):
        super(VAE, self).__init__()
        self.Nc = Nc
        self.Nt = Nt
        self.latent_dim = latent_dim
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


        # 计算编码器输出维度
        self.encoder_output_dim = (Nt // 4)
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar= nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_mu_list = nn.ModuleList([
            nn.Linear(self.encoder_output_dim, latent_dim)
            for _ in range(16 * Nc)
        ])

        self.fc_logvar_list = nn.ModuleList([
            nn.Linear(self.encoder_output_dim, latent_dim)
            for _ in range(16 * Nc)
        ])

        self.transformer_layer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=8*Nc,
                nhead=12,
                dim_feedforward=1024,
                dropout=dropout,
                batch_first=True  # 使用(B, T, C)格式
            ),
            num_layers=2
        )
        self.bi_lstm = nn.LSTM(
            input_size=8 * Nc,
            hidden_size=8 * Nc,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.avgpool = nn.AvgPool1d(kernel_size=2)

        # ----------- Decoder ----------- #
        self.decoder_input = nn.Linear(latent_dim,self.encoder_output_dim)
        self.decoder_input_list = nn.ModuleList([
            nn.Linear(latent_dim,self.encoder_output_dim)
            for _ in range(16 * Nc)
        ])
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

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """编码器前向传播"""
        # x shape: (B, 1, Nc, T)
        x1 = self.spatial_conv(x)  # -> (B, 2Nc, 1, T)
        x2 = self.temp_conv1(x1)  # -> (B, 4Nc, 1, T/2)
        x3 = self.temp_conv2(x2)  # -> (B, 8Nc, 1, T/4)

        # 准备Bi-LSTM输入
        x3_squeezed = x3.squeeze(2).permute(0, 2, 1)  # -> (B, T/4, 8Nc)
        transformer_out, _ = self.bi_lstm(x3_squeezed)  # -> (B, T/4, 16Nc)
        transformer_out = self.avgpool(transformer_out)
        # print(transformer_out.shape)
        transformer_out = transformer_out.permute(0, 2, 1)  # -> (B, 8Nc, T/4)

        e0 = torch.cat([transformer_out, x3.squeeze(2)], dim=1)  # -> (B, 16Nc, T/4)
        # 展平并计算均值和方差
        # e0_flat = e0.view(e0.size(0), -1)
        # mu = self.fc_mu(e0_flat)
        # logvar = self.fc_logvar(e0_flat)
        mu = torch.zeros(e0.size(0), 16 * self.Nc, self.latent_dim, device=e0.device)
        logvar = torch.zeros_like(mu)
        for i in range(16 * self.Nc):
            # 获取第i个通道的特征
            channel_features = e0[:, i, :]  # 形状 (B, T/4)

            # 计算该通道的均值和方差
            mu[:, i, :] = self.fc_mu_list[i](channel_features)
            logvar[:, i, :] = self.fc_logvar_list[i](channel_features)
        """完整前向传播"""
        z = self.reparameterize(mu, logvar)
        """解码器前向传播"""
        B, C, D = z.size()  # 获取输入形状: (B, 16*Nc, latent_dim)

        # 使用列表推导式并行处理所有通道
        x4 = torch.stack([
            self.decoder_input_list[i](z[:, i, :])
            for i in range(16 * self.Nc)
        ], dim=1)  # 形状 (B, 16*Nc, self.encoder_output_dim)

        # 添加高度维度
        x4 = x4.view(B, C, 1, self.Nt // 4)  # 形状 (B, 16*Nc, 1, Nt//4)

        u1 = self.deconv1(x4)  # -> (B, 4Nc, 1, T/2)
        e1 = torch.cat([u1, x2], dim=1)  # -> (B, 8Nc, 1, T/2)
        u2 = self.deconv2(e1)  # -> (B, 2Nc, 1, T)
        e2 = torch.cat([u2, x1], dim=1)  # -> (B, 4Nc, 1, T)
        u3 = self.spatial_deconv(e2)  # -> (B, 1Nc, Nc, T)
        output = self.final_conv(u3)  # -> (B, 1, Nc, T)
        return output, mu, logvar

    def generate(self, num_samples):
        """生成新样本"""
        z = torch.randn(num_samples, self.latent_dim)
        with torch.no_grad():
            return self.decode(z)

    def reconstruct(self, x):
        """重构输入样本"""
        with torch.no_grad():
            mu, _ = self.encode(x)
            return self.decode(mu)
class VAEv2(nn.Module):
    def __init__(self, Nc, Nt, latent_dim=64, dropout=0.5):
        super(VAEv2, self).__init__()
        self.Nc = Nc
        self.Nt = Nt
        self.latent_dim = latent_dim
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


        # 计算编码器输出维度
        # self.encoder_output_dim = 16 * Nc * (Nt // 4)
        #
        # self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        # self.fc_logvar= nn.Linear(self.encoder_output_dim, latent_dim)

        self.transformer_layer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=8*Nc,
                nhead=12,
                dim_feedforward=1024,
                dropout=dropout,
                batch_first=True  # 使用(B, T, C)格式
            ),
            num_layers=4
        )
        self.bi_lstm = nn.LSTM(
            input_size=8 * Nc,
            hidden_size=8 * Nc,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
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
                nn.ConvTranspose2d(4 * Nc, 2 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 0)),
                nn.BatchNorm2d(2 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(4 * Nc, 2 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 1)),
                nn.BatchNorm2d(2 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )

        # Block 7: Spatial Transpose Conv
        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(2 * Nc, 1 * Nc, kernel_size=(Nc, 1), stride=(Nc, 1)),
            nn.BatchNorm2d(1 * Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 8: Final Temporal Conv
        self.final_conv = nn.Conv2d(1 * Nc, 1, kernel_size=(1, 1), stride=(1, 1))

    def encode(self, x):
        """编码器前向传播"""
        # x shape: (B, 1, Nc, T)
        x1 = self.spatial_conv(x)  # -> (B, 2Nc, 1, T)
        x2 = self.temp_conv1(x1)  # -> (B, 4Nc, 1, T/2)
        x3 = self.temp_conv2(x2)  # -> (B, 8Nc, 1, T/4)

        x3_squeezed = x3.squeeze(2).permute(0, 2, 1)  # -> (B, T/4, 8Nc)
        transformer_out = self.transformer_layer(x3_squeezed)  # -> (B, T/4, 16Nc)

        #transformer_out = self.avgpool(transformer_out)
        #print(transformer_out.shape)
        transformer_out =transformer_out.permute(0, 2, 1)  # -> (B, 16Nc, 1, T/4)

        e0 = torch.cat([transformer_out, x3.squeeze(2)], dim=1)  # -> (B, 24Nc, 1, T/4)
        # 展平并计算均值和方差
        mean = torch.mean(e0, dim=-1, keepdim=True)  # 形状变为 (B, C, 1)

        # 2. 计算方差 (无偏估计)
        variance = torch.var(e0, dim=-1, keepdim=True, unbiased=True)  # 形状变为 (B, C, 1)

        # 3. 计算对数方差 (添加小常数避免 log(0))
        log_variance = torch.log(variance + 1e-8)  # 形状保持 (B, C, 1)

        return mean,log_variance,x3.size(-1)

    def reparameterize(self, mu, logvar,T):
        std = torch.exp(0.5 * logvar)  # 形状 (B, C, 1)

        # 2. 扩展 mu 和 std 到目标形状 (B, C, T)
        mu_expanded = mu.expand(-1, -1, T)  # 复制 T 次
        std_expanded = std.expand(-1, -1, T)  # 复制 T 次

        # 3. 生成与扩展后形状相同的随机噪声
        eps = torch.randn_like(std_expanded)  # 形状 (B, C, T)

        # 4. 应用重参数化
        return mu_expanded + eps * std_expanded  # 形状 (B, C, T)

    def decode(self, z):
        """解码器前向传播"""
        # 通过线性层调整维度
        x = z.unsqueeze(2)

        # 解码器前向传播
        u1 = self.deconv1(x)  # -> (B, 4Nc, 1, T/2)
        u2 = self.deconv2(u1)  # -> (B, 2Nc, 1, T)
        u3 = self.spatial_deconv(u2)  # -> (B, 1Nc, Nc, T)
        output = self.final_conv(u3)  # -> (B, 1, Nc, T)

        return output

    def forward(self, x):
        """完整前向传播"""
        mu, logvar,T = self.encode(x)
        z = self.reparameterize(mu, logvar,T)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def generate(self, num_samples):
        """生成新样本"""
        z = torch.randn(num_samples, self.latent_dim)
        with torch.no_grad():
            return self.decode(z)

    def reconstruct(self, x):
        """重构输入样本"""
        with torch.no_grad():
            mu, _ = self.encode(x)
            return self.decode(mu)



class VAEShallowConvNet(nn.Module):
    def __init__(self, n_classes, input_ch, input_time, latent_dim=128, batch_norm=True, batch_norm_alpha=0.1):
        super(VAEShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.input_ch = input_ch
        self.input_time = input_time

        n_ch1 = 40

        # Encoder part (similar to original ShallowConvNet)
        if self.batch_norm:
            self.encoder = nn.Sequential(
                nn.ZeroPad2d(padding=(0, 3, 0, 0)),
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
            )

        # Calculate output dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_ch, input_time)
            out = self.encoder(dummy_input)
            out = torch.square(out)
            out = F.avg_pool2d(out, (1, 75), stride=15)
            self.final_conv_length = out.shape[3]
            self.n_outputs = out.nelement() // out.shape[0]  # Flattened size

        # VAE specific layers
        #print(self.n_outputs)
        self.fc_mu = nn.Linear(self.n_outputs, latent_dim)
        self.fc_logvar = nn.Linear(self.n_outputs, latent_dim)

        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.n_outputs),
            nn.Unflatten(1, (n_ch1, 1, self.final_conv_length)),
            nn.Upsample(scale_factor=(1, 15), mode='nearest'),
            nn.ConvTranspose2d(n_ch1, n_ch1, kernel_size=(1, 75), stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_ch1, 1, kernel_size=(1, 25), stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1, 1, kernel_size=(input_ch, 1), stride=1),
            nn.Sigmoid()  # Ensure output in [0, 1] range
        )

        # Classifier (optional)
        self.clf = None # nn.Linear(latent_dim, n_classes) if n_classes > 0 else None

    def encode(self, x):
        # Encoder forward pass
        x = self.encoder(x)
        x = torch.square(x)
        x = F.avg_pool2d(x, (1, 75), stride=15)
        x = torch.log(torch.clamp(x, min=1e-6))  # Avoid log(0)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)  # Flatten

        # VAE parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Decoder forward pass
        x_recon = self.decoder(z)

        # Crop to match original input size
        _, _, _, recon_time = x_recon.shape
        crop_left = (recon_time - self.input_time) // 2
        crop_right = crop_left + self.input_time
        return x_recon[:, :, :, crop_left:crop_right]

    def forward(self, x):
        # VAE forward pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        # Optional classification
        if self.clf is not None:
            cls_out = self.clf(z)
            return x_recon, mu, logvar, cls_out
        else:
            return x_recon, mu, logvar

    def classify(self, z):
        """Classification using latent space"""
        if self.clf is None:
            raise ValueError("Classifier not initialized")
        return self.clf(z)


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class Stem(nn.Module):
    def __init__(self, data_name, in_planes, out_planes=64, kernel_size=63, patch_size=125, radix=2):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.data_name = data_name
        self.patch_size = patch_size

        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups=radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(Conv(nn.Conv1d(self.out_planes, self.out_planes, kernel_size, 1,
                                             groups=self.out_planes, padding=kernel_size // 2, bias=False),
                                   bn=nn.BatchNorm1d(self.out_planes), activation=None))
            kernel_size //= 2

        self.interFre = InterFre()
        self.downSampling = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)

        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]

        out = self.interFre(out)
        if self.data_name != 'MI1-7' and self.data_name != 'MI1':
            out = out[:, :, :-1]
        out = self.downSampling(out)
        out = self.dp(out)
        return out


class VAE_IFNet(nn.Module):
    def __init__(self, data_name, in_planes, out_planes, kernel_size, radix, patch_size, time_points, latent_dim=64):
        nn.Module.__init__(self)
        self.in_planes = in_planes * radix
        self.out_planes = out_planes
        self.data_name = data_name
        self.patch_size = patch_size
        self.time_points = time_points
        self.latent_dim = latent_dim

        # Encoder stem
        self.stem = Stem(self.data_name, self.in_planes, self.out_planes, kernel_size,
                         patch_size=patch_size, radix=radix)

        # Calculate flattened dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_planes, time_points)
            dummy_output = self.stem(dummy_input)
            self.flatten_dim = dummy_output.numel() // dummy_output.shape[0]

        # VAE layers
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder layers
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)

        # self.decoder_stem = nn.Sequential(
        #     nn.Upsample(scale_factor=patch_size, mode='nearest'),
        #     *[Conv(nn.Conv1d(self.out_planes, self.out_planes, kernel_size // (2 ** i), 1,
        #                      groups=self.out_planes, padding=kernel_size // (2 ** (i + 1)), bias=False),
        #            nn.BatchNorm1d(self.out_planes),
        #            nn.GELU()]
        # for i in range(radix)
        # )
        #
        # self.decoder_sconv = Conv(
        # nn.Conv1d(self.out_planes * radix, self.in_planes, 1, bias=False, groups=radix),
        # bn=nn.BatchNorm1d(self.in_planes),
        # activation=nn.Sigmoid()
        # )
        #
        # self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        # Encoder forward pass
        out = self.stem(x)
        out = out.flatten(1)

        # VAE parameters
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Decoder forward pass
        x = self.decoder_fc(z)
        x = x.view(-1, self.out_planes, self.time_points // self.patch_size)

        # Reverse stem operations
        x = self.decoder_stem(x)

        # Split into radix branches
        branches = torch.split(x, self.out_planes, dim=1)

        # Reverse sconv
        x = self.decoder_sconv(torch.cat(branches, dim=1))

        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def reconstruct(self, x):
        """Reconstruct input without sampling (using mean)"""
        mu, _ = self.encode(x)
        return self.decode(mu)