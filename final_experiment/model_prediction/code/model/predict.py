'''
功能：
1.预测语音情感

最后更新时间：
2024.6.25

负责人：崔彣婧
'''




from tensorflow.keras.models import load_model
from .model import Attention
from .data_utils import co_input_feature
import os
import numpy as np

def predict_emotion():
    emotions={
        0: 'angry',
        1: 'fear',
        2: 'happy',
        3: 'neutral',
        4: 'sad',
        5: 'surprise'
    }
    def predict(model_path):
        # 加载模型
        loaded_model = load_model(model_path, custom_objects={'Attention': Attention})

        # 特征提取
        x=co_input_feature()

        predictions=[]
        # 使用加载的模型进行预测
        for seg in x:
            loaded_y_pred = loaded_model.predict(seg)
            predicted_classes = np.argmax(loaded_y_pred , axis=1)
            predictions.append(int(predicted_classes))
        
        predictions=[emotions[idx] for idx in predictions]

        print(predictions)

        emotion_dict={'angry': 0,'fear': 0,'happy': 0,'neutral': 0,'sad': 0,'surprise': 0 }
        for eomotion in predictions:
            emotion_dict[eomotion]+=1

        fre2emotion={value:key for key,value in emotion_dict.items()}
        max_value = max(emotion_dict.values())
        emotion=fre2emotion[max_value]

        return predictions

    model_path='model_prediction/trained_model1.h5'
    predictions=predict(model_path)
    return predictions

