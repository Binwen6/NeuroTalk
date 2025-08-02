import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset

epsilon = np.finfo(float).eps

# NeuroTalkDataset.py 修改
class myDataset(Dataset):
    def __init__(self, mode, data="./", task="MusicEEG", recon="Y_mel"):
        self.sample_rate = 8000
        self.n_classes = 1  # 音乐重建
        self.mode = mode
        self.savedata = data
        self.task = task
        self.recon = recon
        self.max_audio = 32768.0
        
        # 检查数据目录结构并设置数据长度
        # 首先尝试标准结构 (data/train/Y_mel/)
        if self.mode == 2:
            val_path = self.savedata + '/val/Y_mel/'
        elif self.mode == 1:
            test_path = self.savedata + '/test/Y_mel/'
        else:
            train_path = self.savedata + '/train/Y_mel/'
        
        # 如果标准结构不存在，尝试使用父目录结构
        if self.mode == 2:
            if not os.path.exists(val_path):
                parent_dir = os.path.dirname(self.savedata)
                val_path = parent_dir + '/val/Y_mel/'
            self.lenth = len(os.listdir(val_path))
        elif self.mode == 1:
            if not os.path.exists(test_path):
                parent_dir = os.path.dirname(self.savedata)
                test_path = parent_dir + '/test/Y_mel/'
            self.lenth = len(os.listdir(test_path))
        else:
            if not os.path.exists(train_path):
                parent_dir = os.path.dirname(self.savedata)
                train_path = parent_dir + '/train/Y_mel/'
            self.lenth = len(os.listdir(train_path))
    
    def __getitem__(self, idx):
        # 确定数据目录
        if self.mode == 2:
            forder_name = self.savedata + '/val/'
        elif self.mode == 1:
            forder_name = self.savedata + '/test/'
        else:
            forder_name = self.savedata + '/train/'
        
        # 检查并调整EEG数据路径
        eeg_path = forder_name + self.task + "/"
        if not os.path.exists(eeg_path):
            # 如果task目录不存在，尝试使用父目录结构
            # 例如：如果./dataset/sub1/train/EEG/不存在，尝试./dataset/train/EEG/
            parent_dir = os.path.dirname(self.savedata)
            if self.mode == 2:
                eeg_path = parent_dir + '/val/' + self.task + "/"
            elif self.mode == 1:
                eeg_path = parent_dir + '/test/' + self.task + "/"
            else:
                eeg_path = parent_dir + '/train/' + self.task + "/"
        allFileList = os.listdir(eeg_path)
        allFileList.sort()
        file_name = eeg_path + allFileList[idx]
        input, avg_input, std_input = self.read_data(file_name)
        
        # 检查并调整梅尔频谱数据路径
        mel_path = forder_name + self.recon + "/"
        if not os.path.exists(mel_path):
            # 如果recon目录不存在，尝试使用父目录结构
            if self.mode == 2:
                mel_path = parent_dir + '/val/' + self.recon + "/"
            elif self.mode == 1:
                mel_path = parent_dir + '/test/' + self.recon + "/"
            else:
                mel_path = parent_dir + '/train/' + self.recon + "/"
        allFileList = os.listdir(mel_path)
        allFileList.sort()
        file_name = mel_path + allFileList[idx]
        target, avg_target, std_target = self.read_voice_data(file_name)
        
        # 音乐重建不需要语音相关数据
        voice = torch.zeros((1, 1000), dtype=torch.float32)  # 占位符
        target_cl = torch.zeros((1,), dtype=torch.float32)   # 占位符
        
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        return input, target, target_cl, voice, (avg_target, std_target, avg_input, std_input)
    
    def __len__(self):
        return self.lenth
   
    def read_vector_data(self, file_name,n_classes):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        (r,c) = data.shape
        data = np.reshape(data,(n_classes,r//n_classes,c))
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        
        data   = np.array((data - avg) / std).astype(np.float32)

        return data, avg, std
    
    
    def read_voice_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)
        data = np.array(data).astype(np.float32)
        
        # Ensure consistent shape for mel spectrograms
        if data.shape[0] > 859:  # If more rows than expected, truncate
            data = data[:859, :]
        elif data.shape[0] < 859:  # If fewer rows than expected, pad with zeros
            padding = np.zeros((859 - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        
        data = np.array(data / self.max_audio).astype(np.float32)
        
        # Transpose mel spectrogram to match expected format [mel_bins, time_frames]
        data = data.T  # Now shape is [80, 859]
        
        avg = np.array([0]).astype(np.float32)

        return data, avg, self.max_audio


    def read_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        
        # Ensure consistent shape by padding or truncating
        # Adjust input length to match target output length
        target_length = 859  # Target mel spectrogram length
        if data.shape[0] > target_length:  # If more rows than expected, truncate
            data = data[:target_length, :]
        elif data.shape[0] < target_length:  # If fewer rows than expected, pad with zeros
            padding = np.zeros((target_length - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([data, padding])
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        
        data   = np.array((data - avg) / std).astype(np.float32)

        # Transpose data to match expected input format [channels, time_steps]
        data = data.T  # Now shape is [16, 4976]
            
        return data, avg, std


    def read_raw_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        avg = np.array([0]).astype(np.float32)
        std = np.array([1]).astype(np.float32)

            
        return data, avg, std


