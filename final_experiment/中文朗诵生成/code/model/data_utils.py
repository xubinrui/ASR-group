'''
功能：
1.对文件夹里的数据提取特征并划分数据集
2.对分段后的音频进行数据特征提取，方便后期进行预测

最后更新时间：
2024.6.15

负责人：崔彣婧，赵云路
'''

import numpy as np
import soundfile
import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = r"D:\语音处理\asr1\ASR-group\中文朗诵生成\data\CASIA"  

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        # 如果chroma为True，计算短时傅里叶变换
        if chroma:
            stft = np.abs(librosa.stft(X))
        
        result = np.array([])
        
        # 如果mfcc为True，计算MFCC特征
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        # 如果chroma为True，计算Chroma特征
        if chroma:
            chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feature))
        
        # 如果mel为True，计算Mel spectrogram特征
        if mel:
            mel_feature = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feature))

        return result

# Emotions in the RAVDESS dataset
emotions={
    'angry': 0,
    'fear': 1,
    'happy': 2,
    'neutral': 3,
    'sad': 4,
    'surprise': 5
}

# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    
    for dirname, _, filenames in os.walk(DATA_PATH):
       for filename in filenames:
        file_path = os.path.join(dirname, filename)
        identifiers = filename.split('.')[0].split('-')
        emotion = emotions.get(identifiers[1])
        feature=extract_feature(file_path , mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def co_input_feature():
    x=[]
    
    for dirname, _, filenames in os.walk(r'D:\语音处理\ASR\ASR-group\final_experiment\中文朗诵生成\data\segments'):
       for filename in filenames:
        file_path = os.path.join(dirname, filename)
        feature=extract_feature(file_path , mfcc=True, chroma=True, mel=True)
        x.append(feature)
    return x

# # Split the dataset
# x_train,x_test,y_train,y_test=load_data(test_size=0.25)