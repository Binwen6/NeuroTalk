#!/usr/bin/env python3
"""
NeuroTalk Music Reconstruction Training Script
适配10秒音乐片段的重建训练
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics import MeanSquaredError
import librosa
import soundfile as sf

# 导入NeuroTalk模型
from models.models import Generator, Discriminator
from NeuroTalkDataset import NeuroTalkDataset

class MusicNeuroTalkDataset(NeuroTalkDataset):
    """音乐重建专用数据集"""
    
    def __init__(self, data_dir, config_file="config_music.json"):
        super().__init__(data_dir, config_file)
        
    def __getitem__(self, idx):
        """获取单个数据样本"""
        # 加载EEG数据
        eeg_file = self.eeg_files[idx]
        eeg_data = pd.read_csv(eeg_file, header=None).values.T  # (16, 4975)
        
        # 加载梅尔频谱图
        mel_file = self.mel_files[idx]
        mel_data = pd.read_csv(mel_file, header=None).values.T  # (80, 858)
        
        # 转换为tensor
        eeg_tensor = torch.FloatTensor(eeg_data)
        mel_tensor = torch.FloatTensor(mel_data)
        
        return eeg_tensor, mel_tensor

def train_music_reconstruction():
    """训练音乐重建模型"""
    
    # 加载配置
    with open("config_music.json", "r") as f:
        config = json.load(f)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集
    train_dataset = MusicNeuroTalkDataset("processed_data/train")
    val_dataset = MusicNeuroTalkDataset("processed_data/val")
    test_dataset = MusicNeuroTalkDataset("processed_data/test")
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 创建模型
    generator = Generator(
        input_dim=config["eeg_channels"],
        hidden_dim=256,
        output_dim=config["n_mel_channels"],
        segment_length=config["segment_samples"],
        mel_frames=config["mel_frames"]
    ).to(device)
    
    discriminator = Discriminator(
        input_dim=config["n_mel_channels"],
        hidden_dim=256,
        segment_length=config["mel_frames"]
    ).to(device)
    
    # 损失函数
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # 优化器
    g_optimizer = optim.AdamW(generator.parameters(), lr=1e-4, weight_decay=0.01)
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 训练循环
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        generator.train()
        discriminator.train()
        
        train_loss = 0.0
        for batch_idx, (eeg_batch, mel_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            eeg_batch = eeg_batch.to(device)
            mel_batch = mel_batch.to(device)
            
            # 训练判别器
            d_optimizer.zero_grad()
            
            # 真实样本
            real_output = discriminator(mel_batch)
            real_labels = torch.ones(real_output.size()).to(device)
            d_real_loss = bce_loss(real_output, real_labels)
            
            # 生成样本
            fake_mel = generator(eeg_batch)
            fake_output = discriminator(fake_mel.detach())
            fake_labels = torch.zeros(fake_output.size()).to(device)
            d_fake_loss = bce_loss(fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            g_optimizer.zero_grad()
            
            # 重建损失
            recon_loss = mse_loss(fake_mel, mel_batch)
            
            # 对抗损失
            fake_output = discriminator(fake_mel)
            g_loss = bce_loss(fake_output, real_labels)
            
            # 总损失
            total_g_loss = recon_loss + 0.1 * g_loss
            total_g_loss.backward()
            g_optimizer.step()
            
            train_loss += total_g_loss.item()
        
        # 验证阶段
        generator.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for eeg_batch, mel_batch in val_loader:
                eeg_batch = eeg_batch.to(device)
                mel_batch = mel_batch.to(device)
                
                fake_mel = generator(eeg_batch)
                loss = mse_loss(fake_mel, mel_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), "best_music_generator.pth")
            torch.save(discriminator.state_dict(), "best_music_discriminator.pth")
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")

if __name__ == "__main__":
    train_music_reconstruction() 