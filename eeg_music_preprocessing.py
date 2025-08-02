# eeg_music_preprocessing.py
import mne
import numpy as np
import pandas as pd
import librosa
import os
import glob
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EEGMusicPreprocessor:
    def __init__(self, eeg_dir, music_dir, output_dir):
        self.eeg_dir = eeg_dir
        self.music_dir = music_dir
        self.output_dir = output_dir
        
        # 音频参数
        self.audio_sr = 22050
        self.music_duration = 9.95  # 秒
        
        # 梅尔频谱参数
        self.n_mels = 80
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        
        # EEG参数
        self.eeg_sr = 500
        self.target_eeg_sr = 2500  # NeuroTalk使用的采样率
        self.eeg_channels = 32
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for subdir in ['MusicEEG', 'Y_mel']:
                os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    def load_eeg_data(self, eeg_file):
        """加载BrainVision格式的EEG数据"""
        # 读取.vhdr文件
        raw = mne.io.read_raw_brainvision(eeg_file, preload=True)
        
        # 直接从.vmrk文件读取事件信息
        vmrk_file = eeg_file.replace('.vhdr', '.vmrk')
        events = self.read_vmrk_events(vmrk_file)
        
        # 提取EEG数据
        eeg_data = raw.get_data()
        times = raw.times
        
        return eeg_data, events, raw.info
    
    def read_vmrk_events(self, vmrk_file):
        """手动读取.vmrk文件获取事件信息"""
        events = []
        
        try:
            with open(vmrk_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith('Mk') and 'Stimulus' in line:
                    # 解析事件行，格式：MkX=Stimulus,sY,position,size,channel
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        stimulus_type = parts[1]  # 例如 "s1"
                        position = int(parts[2])  # 时间点位置
                        
                        # 将stimulus类型映射到事件ID
                        stimulus_id = self.map_stimulus_to_id(stimulus_type)
                        
                        if stimulus_id is not None:
                            events.append([position, 0, stimulus_id])
            
            events = np.array(events)
            print(f"从.vmrk文件读取到 {len(events)} 个事件")
            
            # 调试：打印每个stimulus的事件数量
            stimulus_counts = {}
            for event in events:
                event_id = event[2]
                if event_id in [10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009]:
                    stimulus_name = f"s{event_id - 10001}"
                    if stimulus_name not in stimulus_counts:
                        stimulus_counts[stimulus_name] = 0
                    stimulus_counts[stimulus_name] += 1
            
            print(f"各stimulus事件数量: {stimulus_counts}")
            
            return events
            
        except Exception as e:
            print(f"读取.vmrk文件时出错: {e}")
            return np.array([])
    
    def map_stimulus_to_id(self, stimulus_type):
        """将stimulus类型映射到事件ID"""
        mapping = {
            's1': 10002,
            's2': 10003,
            's3': 10004,
            's4': 10005,
            's5': 10006,
            's6': 10007,
            's7': 10008,
            's8': 10009
        }
        return mapping.get(stimulus_type)
    
    def extract_stimulus_segments(self, eeg_data, events, stimulus_id):
        """提取特定stimulus的EEG片段"""
        # 找到该stimulus的所有事件
        stimulus_events = events[events[:, 2] == stimulus_id]
        
        if len(stimulus_events) == 0:
            return []
        
        # 只取前5次有效事件（根据您的描述，每个stimulus播放5次）
        # 即使有更多重复事件，也只取前5次
        valid_events = stimulus_events[:5]
        
        print(f"    找到 {len(stimulus_events)} 个事件，使用前 {len(valid_events)} 个有效事件")
        
        segments = []
        sampling_rate = 500  # 您的EEG采样率
        segment_duration = 9.95  # 音频时长
        segment_samples = int(sampling_rate * segment_duration)
        
        for i, event in enumerate(valid_events):
            start_sample = event[0]  # 事件开始时间点
            
            # 检查是否有足够的数据
            if start_sample + segment_samples > eeg_data.shape[1]:
                print(f"    警告: 事件 {event} 超出EEG数据范围")
                continue
            
            # 提取EEG片段（保持10秒完整性）
            eeg_segment = eeg_data[:, start_sample:start_sample + segment_samples]
            
            # 预处理EEG片段
            eeg_processed = self.preprocess_eeg_segment(eeg_segment)
            
            # 确保EEG片段是16通道
            if eeg_processed.shape[0] == 32:
                # 选择16个主要通道
                selected_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                eeg_processed = eeg_processed[selected_channels, :]
            
            segments.append(eeg_processed)
        
        return segments
    
    def preprocess_eeg_segment(self, eeg_segment):
        """预处理EEG片段"""
        # 1. 带通滤波 (30-120Hz)
        b, a = signal.butter(5, [30, 120], btype='band', fs=500)
        eeg_filtered = signal.filtfilt(b, a, eeg_segment, axis=1)
        
        # 2. 陷波滤波 (60Hz)
        b_notch, a_notch = signal.iirnotch(60, 30, fs=500)
        eeg_notched = signal.filtfilt(b_notch, a_notch, eeg_filtered, axis=1)
        
        # 3. 基线校正 (减去前500ms的平均值)
        baseline_samples = int(0.5 * 500)  # 500ms
        if eeg_notched.shape[1] > baseline_samples:
            baseline = np.mean(eeg_notched[:, :baseline_samples], axis=1, keepdims=True)
            eeg_baseline_corrected = eeg_notched - baseline
        else:
            eeg_baseline_corrected = eeg_notched
        
        # 4. 标准化
        eeg_normalized = (eeg_baseline_corrected - np.mean(eeg_baseline_corrected, axis=1, keepdims=True)) / (np.std(eeg_baseline_corrected, axis=1, keepdims=True) + 1e-8)
        
        return eeg_normalized
    
    def extract_mel_spectrogram(self, audio_file):
        """提取梅尔频谱图"""
        # 加载音频
        audio, sr = librosa.load(audio_file, sr=22050)
        
        # 计算梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0,
            fmax=8000
        )
        
        # 转换为对数尺度
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def save_segment(self, eeg_data, mel_data, filename):
        """保存数据片段"""
        # 保存EEG数据
        eeg_file = os.path.join(self.output_dir, 'MusicEEG', filename + '_eeg.csv')
        pd.DataFrame(eeg_data.T).to_csv(eeg_file, index=False, header=False)
        
        # 保存梅尔频谱数据
        mel_file = os.path.join(self.output_dir, 'Y_mel', filename + '_mel.csv')
        pd.DataFrame(mel_data.T).to_csv(mel_file, index=False, header=False)
    
    def process_subject_data(self, subject_name):
        """处理单个被试的数据"""
        print(f"处理被试: {subject_name}")
        
        # 查找EEG文件
        eeg_pattern = os.path.join(self.eeg_dir, f"{subject_name}*.vhdr")
        eeg_files = glob.glob(eeg_pattern)
        
        if not eeg_files:
            print(f"未找到被试 {subject_name} 的EEG文件")
            return []
        
        eeg_file = eeg_files[0]
        print(f"找到EEG文件: {eeg_file}")
        
        try:
            # 加载EEG数据
            eeg_data, events, info = self.load_eeg_data(eeg_file)
            print(f"EEG数据形状: {eeg_data.shape}")
            print(f"事件数量: {len(events)}")
            
            if len(events) == 0:
                print(f"警告: 被试 {subject_name} 没有找到任何事件")
                return []
            
        except Exception as e:
            print(f"加载被试 {subject_name} 的EEG数据时出错: {e}")
            return []
        
        # 音频文件映射 - 使用实际的文件名
        audio_files = {
            's1': 'All Of Me - All of Me (Karaoke Version).wav',
            's2': 'Richard Clayderman - 梦中的鸟.wav', 
            's3': 'Robin Spielberg - Turn the Page.wav',
            's4': 'dylanf - 梦中的婚礼 (经典钢琴版).wav',
            's5': '文武贝 - 夜的钢琴曲5.wav',
            's6': '昼夜 - 千与千寻 (钢琴版).wav',
            's7': '演奏曲 - 【钢琴Piano】雨中漫步Stepping On The Rainy S.wav',
            's8': '郭宴 - 天空之城 (钢琴版).wav'
        }
        
        # 事件ID映射
        stimulus_ids = {
            's1': 10002,
            's2': 10003,
            's3': 10004,
            's4': 10005,
            's5': 10006,
            's6': 10007,
            's7': 10008,
            's8': 10009
        }
        
        # 获取该被试实际有的stimulus类型
        available_stimuli = self.get_available_stimuli(events)
        print(f"  该被试有的stimulus类型: {list(available_stimuli.keys())}")
        
        segments = []
        
        # 只处理该被试实际有的stimulus
        for stimulus_name, stimulus_id in stimulus_ids.items():
            if stimulus_name not in available_stimuli:
                print(f"  跳过 {stimulus_name} - 该被试没有此stimulus")
                continue
                
            print(f"  处理stimulus: {stimulus_name} (ID: {stimulus_id})")
            
            # 检查该被试是否有这个stimulus的事件
            stimulus_events = events[events[:, 2] == stimulus_id]
            print(f"    事件ID {stimulus_id} 找到 {len(stimulus_events)} 个事件")
            
            if len(stimulus_events) == 0:
                print(f"    警告: 被试 {subject_name} 没有找到 {stimulus_name} 的事件")
                continue
            
            # 提取EEG片段
            eeg_segments = self.extract_stimulus_segments(eeg_data, events, stimulus_id)
            print(f"    找到 {len(eeg_segments)} 个EEG片段")
            
            if len(eeg_segments) == 0:
                print(f"    警告: 没有提取到有效的EEG片段")
                continue
            
            # 加载对应的音频文件
            audio_file = os.path.join(self.music_dir, audio_files[stimulus_name])
            if not os.path.exists(audio_file):
                print(f"    警告: 音频文件不存在: {audio_file}")
                continue
                
            # 生成梅尔频谱图
            mel_spectrogram = self.extract_mel_spectrogram(audio_file)
            print(f"    梅尔频谱图形状: {mel_spectrogram.shape}")
            
            # 创建数据片段
            for i, eeg_segment in enumerate(eeg_segments):
                segment = {
                    'subject': subject_name,
                    'stimulus': stimulus_name,
                    'eeg_data': eeg_segment,
                    'mel_spectrogram': mel_spectrogram,
                    'segment_id': f"{subject_name}_{stimulus_name}_{i+1}"
                }
                segments.append(segment)
        
        print(f"被试 {subject_name} 处理完成，共生成 {len(segments)} 个数据片段")
        return segments
    
    def get_available_stimuli(self, events):
        """获取该被试实际有的stimulus类型"""
        available = {}
        
        # 统计每个stimulus的事件数量
        stimulus_counts = {}
        for event in events:
            event_id = event[2]
            if event_id in [10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009]:
                stimulus_name = f"s{event_id - 10001}"
                if stimulus_name not in stimulus_counts:
                    stimulus_counts[stimulus_name] = 0
                stimulus_counts[stimulus_name] += 1
        
        # 只保留有足够事件的stimulus（至少5次）
        for stimulus_name, count in stimulus_counts.items():
            if count >= 5:  # 每个stimulus至少需要5次有效事件
                available[stimulus_name] = count
        
        return available
    
    def split_dataset(self, all_segments, train_ratio=0.7, val_ratio=0.15):
        """划分数据集"""
        np.random.shuffle(all_segments)
        
        total_segments = len(all_segments)
        train_end = int(total_segments * train_ratio)
        val_end = int(total_segments * (train_ratio + val_ratio))
        
        train_segments = all_segments[:train_end]
        val_segments = all_segments[train_end:val_end]
        test_segments = all_segments[val_end:]
        
        return train_segments, val_segments, test_segments
    
    def save_dataset(self, segments, dataset_type):
        """保存数据集"""
        print(f"保存 {dataset_type} 数据集，共 {len(segments)} 个片段")
        
        # 创建输出目录
        eeg_dir = os.path.join(self.output_dir, dataset_type, 'EEG')
        mel_dir = os.path.join(self.output_dir, dataset_type, 'Y_mel')
        os.makedirs(eeg_dir, exist_ok=True)
        os.makedirs(mel_dir, exist_ok=True)
        
        for i, segment in enumerate(segments):
            # 保存EEG数据
            eeg_filename = f"{segment['segment_id']}.csv"
            eeg_path = os.path.join(eeg_dir, eeg_filename)
            
            # 确保EEG数据是16通道
            eeg_data = segment['eeg_data']
            if eeg_data.shape[0] != 16:
                # 如果还是32通道，选择前16个
                eeg_data = eeg_data[:16, :]
            
            # 保存为CSV格式
            eeg_df = pd.DataFrame(eeg_data.T)
            eeg_df.to_csv(eeg_path, index=False, header=False)
            
            # 保存梅尔频谱图
            mel_filename = f"{segment['segment_id']}_mel.csv"
            mel_path = os.path.join(mel_dir, mel_filename)
            
            mel_data = segment['mel_spectrogram']
            mel_df = pd.DataFrame(mel_data.T)
            mel_df.to_csv(mel_path, index=False, header=False)
    
    def process_all_data(self):
        """处理所有数据"""
        # 被试列表
        subjects = [
            'jia_haoxuan', 'zhang_geng', 'zhang_yichi',  # 完整组
            'Shi_yuxin', 'wang_yingshan3', 'xiao_xingtong',  # 部分组
            'jin_shibo', 'wu_zhaowei2', 'ouyang_jingqian', 'wang_yingshan2', 'zheng_kaizhong2'  # 最小组
        ]
        
        all_segments = []
        
        # 处理每个被试
        for subject in subjects:
            try:
                segments = self.process_subject_data(subject)
                if segments:
                    all_segments.extend(segments)
                    print(f"被试 {subject} 添加了 {len(segments)} 个片段")
                else:
                    print(f"被试 {subject} 没有生成任何片段")
            except Exception as e:
                print(f"处理被试 {subject} 时出错: {e}")
                continue
        
        if not all_segments:
            print("错误: 没有生成任何数据片段！")
            return
        
        print(f"总共生成了 {len(all_segments)} 个数据片段")
        
        # 划分数据集
        train_segments, val_segments, test_segments = self.split_dataset(all_segments)
        
        # 保存数据集
        self.save_dataset(train_segments, 'train')
        self.save_dataset(val_segments, 'val')
        self.save_dataset(test_segments, 'test')
        
        print(f"数据预处理完成！")
        print(f"训练集: {len(train_segments)} 个片段")
        print(f"验证集: {len(val_segments)} 个片段")
        print(f"测试集: {len(test_segments)} 个片段")

if __name__ == "__main__":
    # 运行预处理脚本
    preprocessor = EEGMusicPreprocessor(
        eeg_dir='/root/autodl-tmp/Generate',
        music_dir='/root/autodl-tmp/music_gen_wav_22050',
        output_dir='/root/autodl-tmp/dataset'
    )
    preprocessor.process_all_data()