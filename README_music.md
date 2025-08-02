# NeuroTalk 音乐重建训练指南

## 概述
本指南说明如何使用修改后的NeuroTalk框架进行EEG到音乐的重建训练。

## 数据预处理
首先运行数据预处理脚本：
```bash
python eeg_music_preprocessing.py
```

这将生成以下数据结构：
```
dataset/
├── train/
│   ├── EEG/          # 训练集EEG数据
│   └── Y_mel/        # 训练集梅尔频谱图
├── val/
│   ├── EEG/          # 验证集EEG数据
│   └── Y_mel/        # 验证集梅尔频谱图
└── test/
    ├── EEG/           # 测试集EEG数据
    └── Y_mel/         # 测试集梅尔频谱图
```

## 音乐重建训练

### 1. 基本训练命令
```bash
python train.py --music_mode True --batch_size 8 --max_epochs 100
```

### 2. 参数说明
- `--music_mode True`: 启用音乐重建模式
- `--batch_size 8`: 批次大小（根据GPU内存调整）
- `--max_epochs 100`: 最大训练轮数
- `--lr_g 1e-4`: 生成器学习率
- `--lr_d 1e-4`: 判别器学习率
- `--gpuNum [0]`: 使用的GPU编号

### 3. 完整训练命令示例
```bash
python train.py \
    --music_mode True \
    --batch_size 8 \
    --max_epochs 100 \
    --lr_g 1e-4 \
    --lr_d 1e-4 \
    --gpuNum [0] \
    --logDir ./TrainResult_music \
    --sub music_reconstruction \
    --task MusicEEG
```

## 训练过程

### 1. 数据加载
- 训练集：约227个10秒EEG-音乐对
- 验证集：约49个10秒EEG-音乐对
- 测试集：约49个10秒EEG-音乐对

### 2. 模型架构
- **生成器**：EEG (16, 4975) → 梅尔频谱图 (80, 858)
- **判别器**：梅尔频谱图 (80, 858) → 真实/伪造概率

### 3. 损失函数
- **重建损失**：MSE损失，衡量重建梅尔频谱图与目标的差异
- **对抗损失**：BCE损失，用于GAN训练
- **总损失**：重建损失 + 0.1 × 对抗损失

### 4. 训练监控
训练过程中会显示：
- 每个epoch的训练损失和验证损失
- 学习率变化
- 最佳模型保存信息

## 模型保存
训练过程中会自动保存：
- `checkpoint_g.pt`: 生成器检查点
- `BEST_checkpoint_g.pt`: 最佳生成器模型
- `checkpoint_d.pt`: 判别器检查点
- `BEST_checkpoint_d.pt`: 最佳判别器模型

## 推理测试
训练完成后，使用测试集评估模型：
```bash
python eval.py \
    --model_path ./TrainResult_music/music_reconstruction/MusicEEG/savemodel/BEST_checkpoint_g.pt \
    --test_data ./dataset/test/
```

## 注意事项

1. **GPU内存**：如果GPU内存不足，减小batch_size
2. **训练时间**：完整训练可能需要几小时到一天
3. **数据路径**：确保EEG和音频文件路径正确
4. **配置文件**：音乐模式会自动使用`config_music.json`和相关模型配置

## 与语音重建的区别

| 特性 | 语音重建 | 音乐重建 |
|------|----------|----------|
| 数据长度 | 2秒片段 | 10秒完整片段 |
| 数据量 | 约1000个片段 | 约325个片段 |
| 类别数 | 13个语音类别 | 8个音乐类别 |
| 处理逻辑 | 复杂（unseen/seen分离） | 简化（直接训练） |
| 评估指标 | CER, MOS | RMSE, 主观评估 |

## 故障排除

1. **CUDA内存不足**：减小batch_size或使用更少的GPU
2. **数据加载错误**：检查数据路径和文件格式
3. **模型配置错误**：确保配置文件存在且格式正确
4. **训练不收敛**：调整学习率或损失权重 