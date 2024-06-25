'''
功能：
1.对文件夹里的数据提取特征并划分数据集
2.对分段后的音频进行数据特征提取，方便后期进行预测

最后更新时间：
2024.6.25

负责人：崔彣婧，赵云路
'''
import numpy as np
import soundfile
import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = r"././data/CASIA" 

def get_spectrogram(file_name):
    # Load the audio file
    y, sr = librosa.load(file_name)
    
    # Compute the Short-time Fourier transform (STFT)
    D = librosa.stft(y)
    
    # Convert the complex values to magnitude and phase
    S, phase = librosa.magphase(D)
    
    # Convert amplitude to power (square)
    S_power = S**2
    
    # Convert power spectrogram to decibel units
    S_db = librosa.power_to_db(S_power, ref=np.max)

    return S_db, sr  # Return the spectrogram in dB and the sample rate

EMOTIONS = {
    'angry': 0,
    'fear': 1,
    'happy': 2,
    'neutral': 3,
    'sad': 4,
    'surprise': 5
}

DATA_PATH = r"C:\Users\HP\OneDrive\桌面\语音信息处理\tts\CASIA\CASIA\6\sad"

data_list = []

for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        identifiers = filename.split('.')[0].split('-')

        emotion = EMOTIONS.get(identifiers[1])  # 使用get方法以防万一有未知的标签
        feature, sr = get_spectrogram(file_path)

        # 将提取的信息作为字典添加到data_list列表中
        data_list.append({
            "Emotion": emotion,
            "Path": file_path,
            "Feature": feature,
            "SampleRate": sr
        })

# 如果需要将数据保存到DataFrame
data = pd.DataFrame(data_list)

# 找到形状最大的频谱图
max_shape = max(data['feature'], key=lambda x: x.shape[1]).shape

# 填充其他频谱图以匹配最大形状
padded_features = [np.pad(feature, pad_width=((0, 0), (0, max_shape[1] - feature.shape[1])), mode='constant', constant_values=0) for feature in data['feature']]

# 将填充后的频谱图作为特征
X = np.array(padded_features)

# 将情绪标签作为目标
y = np.array(data['Emotion'].tolist())

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 检查分割后的数据形状
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


