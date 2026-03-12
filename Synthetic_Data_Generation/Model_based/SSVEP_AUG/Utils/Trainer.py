import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
from Models import Generator, Discriminator
from etc.global_config import config


def build_generator(arch, Nc, Nt):
    """Build generator based on architecture name."""
    if arch == 'CNN':
        return Generator.EEGDenoiseGenerator_NoSeq(Nc)
    elif arch == 'CNN_LSTM':
        return Generator.EEGDenoiseGenerator(Nc, Nt)
    elif arch == 'CNN_Trans':
        return Generator.EEGDenoiseGeneratorv2(Nc, Nt)
    else:
        raise ValueError(f"Unknown generator architecture: {arch}")


def train_aug_generator(clean_data, labels, arch, loss_type, info, save_path):
    """
    Train a generator for model-based data augmentation.
    Uses the original data directly (no noise augmentation on dataset).

    Three loss variants:
      - 'base': reconstruction (L1) + classification (CE)
      - 'gan':  reconstruction (L1) + classification (CE) + adversarial (GAN)
      - 'vae':  reconstruction (L1) + classification (CE) + KL divergence

    Three architectures:
      - 'CNN':       EEGDenoiseGenerator_NoSeq (pure CNN)
      - 'CNN_LSTM':  EEGDenoiseGenerator (CNN + Bi-LSTM bottleneck)
      - 'CNN_Trans': EEGDenoiseGeneratorv2 (CNN + Transformer bottleneck)

    Args:
        clean_data: (N, Nc, T) numpy array of clean EEG signals
        labels: (N,) numpy array of class labels
        arch: 'CNN', 'CNN_TRANS', or 'CNN_LSTM'
        loss_type: 'base', 'gan', or 'vae'
        info: [kfold, datasetid, subject, arch_name, _, device_str]
        save_path: path to save the trained generator model
    """
    device = info[3]
    datasetid = config["train_param"]["datasets"]
    Nc = clean_data.shape[1]
    Nt = clean_data.shape[2]
    Nf = config[f"data_param{datasetid}"]["Nf"]

    epochs = config["train_param"]["epochs"]
    bz = config["train_param"]["bz"]
    lr_G = config["gan_model_param"]["lr_G"]
    lr_D = config["gan_model_param"]["lr_D"]
    wd_G = config["gan_model_param"]["wd_G"]
    wd_D = config["gan_model_param"]["wd_D"]
    lambda_G = config["gan_model_param"]["lambda_G"]
    lambda_D = config["gan_model_param"]["lambda_D"]
    lambda_vae = config["gan_model_param"]["lambda_vae"]
    lambda_kl = config["gan_model_param"]["lambda_kl"]
    lr_jitter = config["gan_model_param"]["lr_jitter"]

    # Build models
    generator = build_generator(arch, Nc, Nt).to(device)
    discriminator = Discriminator.Spectral_Normalization(
        Discriminator.ESNet(Nc, Nt, Nf)
    ).to(device)

    # Dataset & DataLoader — use original data directly, (x, label)
    data_tensor = torch.tensor(clean_data, dtype=torch.float32).unsqueeze(1)  # (N, 1, Nc, T)
    label_tensor = torch.tensor(labels.flatten(), dtype=torch.long)
    dataset = TensorDataset(data_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=bz, shuffle=True, drop_last=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, weight_decay=wd_G)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, weight_decay=wd_D)

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_G, T_max=epochs * max(len(dataloader), 1), eta_min=5e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_D, T_max=epochs * max(len(dataloader), 1), eta_min=5e-6)

    criterion_ce = nn.CrossEntropyLoss()
    log = lambda x: torch.log(x + 1e-8)

    use_adversarial = (loss_type == 'gan')
    use_kl = (loss_type == 'vae')
    is_log_variance = (arch == 'CNN_TRANS')

    print(f"Training {arch}-{loss_type} generator (epochs={epochs})...")

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        n_batches = 0

        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            # ========================
            # 1. Train Discriminator
            # ========================
            optimizer_D.zero_grad()

            with torch.no_grad():
                _, _, _, _, gen_x = generator(X)

            d_real = discriminator(X)
            d_fake = discriminator(gen_x.detach())

            # D classification loss
            L_Dfeat = -log(d_real['cls'][range(len(y)), y]).mean() - \
                       log(d_fake['cls'][range(len(y)), y]).mean()

            if use_adversarial:
                L_Dadv = -log(d_real['adv']).mean() - log(1 - d_fake['adv']).mean()
                loss_D = L_Dadv + lambda_D * L_Dfeat
            else:
                loss_D = L_Dfeat

            loss_D.backward()
            optimizer_D.step()
            if lr_jitter:
                scheduler_D.step()

            # ========================
            # 2. Train Generator
            # ========================
            optimizer_G.zero_grad()

            out1, out2, out3, out4, gen_x = generator(X)

            d_gen = discriminator(gen_x)

            # Classification loss
            L_ca = criterion_ce(d_gen['cls'], y)

            # Reconstruction loss (L1)
            recon_loss = F.l1_loss(gen_x, X, reduction='mean')

            # Base loss: cls + reconstruction
            loss_G = lambda_G * L_ca + lambda_vae * recon_loss

            # Adversarial loss (GAN only)
            if use_adversarial:
                L_Gadv = -log(d_gen['adv']).mean()
                loss_G = loss_G + L_Gadv

            # KL divergence loss (VAE only)
            if use_kl:
                if is_log_variance:
                    log_var1 = out2
                    log_var2 = out4
                else:
                    log_var1 = torch.log(out2 + 1e-8)
                    log_var2 = torch.log(out4 + 1e-8)

                # Standard VAE KL: -0.5 * (1 + log_var - mu^2 - exp(log_var))
                kl_loss1 = -0.5 * torch.mean(1 + log_var1 - out1.pow(2) - log_var1.exp())
                kl_loss2 = -0.5 * torch.mean(1 + log_var2 - out3.pow(2) - log_var2.exp())
                kl_loss = kl_loss1 + kl_loss2
                loss_G = loss_G + lambda_kl * kl_loss

            loss_G.backward()
            optimizer_G.step()
            if lr_jitter:
                scheduler_G.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_G = epoch_loss_G / max(n_batches, 1)
            avg_D = epoch_loss_D / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs} | G Loss: {avg_G:.4f} | D Loss: {avg_D:.4f}")

    # Save the full generator model (torch.save saves the entire model for torch.load compatibility)
    torch.save(generator, save_path)
    print(f"Generator saved to {save_path}")
    print(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    return generator