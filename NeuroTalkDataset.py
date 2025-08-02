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
        
        # 根据模式设置数据长度
        if self.mode == 2:
            self.lenth = len(os.listdir(self.savedata + '/val/Y_mel/'))
        elif self.mode == 1:
            self.lenth = len(os.listdir(self.savedata + '/test/Y_mel/'))
        else:
            self.lenth = len(os.listdir(self.savedata + '/train/Y_mel/'))
    
    def __getitem__(self, idx):
        # 确定数据目录
        if self.mode == 2:
            forder_name = self.savedata + '/val/'
        elif self.mode == 1:
            forder_name = self.savedata + '/test/'
        else:
            forder_name = self.savedata + '/train/'
        
        # 读取EEG数据
        allFileList = os.listdir(forder_name + self.task + "/")
        allFileList.sort()
        file_name = forder_name + self.task + '/' + allFileList[idx]
        input, avg_input, std_input = self.read_data(file_name)
        
        # 读取梅尔频谱数据
        allFileList = os.listdir(forder_name + self.recon + "/")
        allFileList.sort()
        file_name = forder_name + self.recon + '/' + allFileList[idx]
        target, avg_target, std_target = self.read_data(file_name)
        
        # 音乐重建不需要语音相关数据
        voice = torch.zeros((1, 1000))  # 占位符
        target_cl = torch.zeros((1,))   # 占位符
        
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        return input, target, target_cl, voice, (avg_target, std_target, avg_input, std_input)
   
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
        
        data = np.array(data / self.max_audio).astype(np.float32)
        avg = np.array([0]).astype(np.float32)

        return data, avg, self.max_audio


    def read_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)
        
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = (max_ - min_) / 2
        
        data   = np.array((data - avg) / std).astype(np.float32)

            
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


