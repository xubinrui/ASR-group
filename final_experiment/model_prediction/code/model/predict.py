'''
功能：
1.预测语音情感

最后更新时间：
2024.6.15

负责人：崔彣婧
'''
from tensorflow.keras.models import load_model
from .model import Attention
from .data_utils import co_input_feature
import os

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
        

    # 返回一个编译好的模型
    # 与保存模型相同的配置
        loaded_model = load_model(model_path, custom_objects={'Attention': Attention})


        x=co_input_feature()

        # 使用加载的模型进行预测
        loaded_y_pred = loaded_model.predict(x)

        predictions=[emotions[idx] for idx in loaded_y_pred]

        print(predictions)
        emotion_dict={'angry': 0,'fear': 0,'happy': 0,'neutral': 0,'sad': 0,'surprise': 0 }
        for eomotion in predictions:
            emotion_dict[eomotion]+=1

        fre2emotion={value:key for key,value in emotion_dict.items()}
        max_value = max(emotion_dict.values())
        emotion=fre2emotion[max_value]

        return predictions

    model_path='model_prediction\my_model.h5'
    predictions=predict(model_path)
    return predictions

