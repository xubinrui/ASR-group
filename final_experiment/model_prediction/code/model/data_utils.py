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
    # 加载音频文件
    y, sr = librosa.load(file_name)
    
    # 计算短时傅里叶变换 (STFT)
    D = librosa.stft(y)
    
    # 将复数值转换为幅度和相位
    S, phase = librosa.magphase(D)
    
    # 将幅度转换为功率 (平方)
    S_power = S**2
    
    # 将功率谱图转换为分贝单位
    S_db = librosa.power_to_db(S_power, ref=np.max)

    # 返回分贝单位的谱图和采样率
    return S_db, sr

EMOTIONS = {
    'angry': 0,
    'fear': 1,
    'happy': 2,
    'neutral': 3,
    'sad': 4,
    'surprise': 5
}

def process_audio_files(DATA_PATH, target_length=150, test_size=0.25, random_state=42):
    data_list = []

    for dirname, _, filenames in os.walk(DATA_PATH):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            identifiers = filename.split('.')[0].split('-')

            emotion = EMOTIONS.get(identifiers[1], -1)  # 使用get方法以防万一有未知的标签
            if emotion != -1:
                feature, sr = get_spectrogram(file_path)

                # 将提取的信息作为字典添加到data_list列表中
                data_list.append({
                    "Emotion": emotion,
                    "Path": file_path,
                    "Feature": feature,
                    "SampleRate": sr
                })

    # 将data_list转换为DataFrame
    data = pd.DataFrame(data_list)

    # 找到形状最大的频谱图
    max_shape = max(data['Feature'], key=lambda x: x.shape[1]).shape

    # 填充或截断频谱图以匹配目标长度
    padded_features = []
    for feature in data['Feature']:
        if feature.shape[1] > target_length:
            # 截断长的频谱图
            padded_feature = feature[:, :target_length]
        else:
            # 填充短的频谱图
            padded_feature = np.pad(feature, pad_width=((0, 0), (0, target_length - feature.shape[1])), mode='constant', constant_values=0)
        padded_features.append(padded_feature)

    # 将填充后的频谱图作为特征
    X = np.array(padded_features)

    # 将情绪标签作为目标
    y = np.array(data['Emotion'].tolist())

    X = np.expand_dims(X, axis=-1)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# 调用函数并进行数据处理
X_train, X_test, y_train, y_test = process_audio_files(DATA_PATH)


#提取所有输入文件的特征
def co_input_feature():
    x=[]
    
    for dirname, _, filenames in os.walk('segments'):
       for filename in filenames:
        file_path = os.path.join(dirname, filename)
        feature,_=get_spectrogram(file_path)
        feature = np.array(feature)
        if feature.shape[1] > 150:
            padded_feature = feature[:, :150]
        else:
            padded_feature = np.pad(feature, pad_width=((0, 0), (0, 150 - feature.shape[1])), mode='constant', constant_values=0)
        padded_feature = np.expand_dims(padded_feature, axis=-1)
        X = np.expand_dims(padded_feature, axis=0)
        # print(X.shape)
        x.append(X)
        
    return x
# feature=co_input_feature()
