'''
功能：
1.预测语音情感

最后更新时间：
2024.6.15

负责人：崔彣婧
'''


from joblib import load
from .data_utils import co_input_feature
import os


# x=[]
# for filename in ['././data/CASIA/fear/203-fear-zhaoquanyin.wav']:
#     feature=extract_feature(filename , mfcc=True, chroma=True, mel=True)
#     x.append(feature)
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
        loaded_model = load(model_path)

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

    model_path=r'D:\语音处理\asr1\ASR-group\中文朗诵生成\mlp_model.joblib'
    predictions=predict(model_path)
    return predictions

