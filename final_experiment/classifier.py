'''
功能：进行情感分类并合成音频
负责人：徐彬芮、崔彣婧
更新时间：2024.6.25

'''
import soundfile as sf
import numpy as np
from collections import Counter
from model_prediction.code.utils.vocie_segment import co_voiceseg, save_segmented_audio
from model_prediction.code.model.predict import predict_emotion
from mix import mix_audio_files

# 获取背景音乐文件路径
def get_background_music_path(emotion):
    emotion_to_music = {
        'happy': 'wav_document_wav/happy.wav',
        'sad': 'wav_document_wav/sad.wav',
        'angry': 'wav_document_wav/anger.wav',
        'fear': 'wav_document_wav/fear.wav',
        'neutral': 'wav_document_wav/neutral.wav',
        'surprise': 'wav_document_wav/happy.wav'
    }
    return emotion_to_music.get(emotion, None)

def classifier(text=None, denoised_audio_path=None, wlen=7):
    '''
    wlen为判断语音情感的窗口大小，调整越小，音频的背景音乐转换也许会越频繁
    '''
    if text is not None:
        file_path = 'wav_document/demo.wav'
        print('text:', text)
    else:
        # 获取分段语音（可以使去噪后的，也可以是原始语音）
        voiceseg = co_voiceseg(denoised_audio_path)
        # 保存分段后的语音
        save_segmented_audio(denoised_audio_path, voiceseg)
        # 预测情感
        emotions = predict_emotion()
        print('emotions:', emotions)
        
        # 加载原始音频
        original_audio, rate = sf.read(denoised_audio_path)
        segment_duration = len(original_audio) // len(emotions)
        temp=None
        combined_audio = np.array([])
      
        
        start_time=0
        end_time=segment_duration
        for i in range(len(emotions)):
          
            # start_time = i * segment_duration
            # end_time = start_time + segment_duration
            # if end_time > len(original_audio):
            #     end_time = len(original_audio)

            # 获取当前片段的情感
            window_emotions = emotions[max(0, i-wlen):i+1]
            most_common_emotion = Counter(window_emotions).most_common(1)[0][0]
            if temp!=most_common_emotion:
                temp = most_common_emotion
                # 获取该段的音频片段
                audio_segment = original_audio[start_time:end_time]
                print('start_time:', start_time)
                print('end_time:', end_time)
                # 获取对应情感的背景音乐
                background_music_path = get_background_music_path(most_common_emotion)
                if background_music_path is not None:
                    # 混合音频片段和背景音乐
                    background_music, _ = sf.read(background_music_path)
                    background_music = background_music[:len(audio_segment)]
                    if len(background_music.shape) > 1:
                        background_music = background_music.mean(axis=1)
                
                    
                    mixed_segment = mix_audio_files(audio_segment, background_music)
                    start_time =(i+1) * segment_duration
                    end_time += segment_duration
                    if end_time > len(original_audio):
                        end_time = len(original_audio)
                else:
                    mixed_segment = audio_segment
                
                combined_audio = np.concatenate((combined_audio, mixed_segment))
                print('combined_audio:', combined_audio.shape)
                print('当前情感：', most_common_emotion)
            else:
                end_time += segment_duration
                if end_time > len(original_audio):
                    end_time = len(original_audio)
                    print(len(original_audio))
                    print('start_time1:', start_time)
                    print('end_time1:', end_time)
                    audio_segment = original_audio[start_time:end_time]
                    background_music_path = get_background_music_path(most_common_emotion)
                    if background_music_path is not None:
                        # 混合音频片段和背景音乐
                        background_music, _ = sf.read(background_music_path)
                        background_music = background_music[:len(audio_segment)]
                        if len(background_music.shape) > 1:
                            background_music = background_music.mean(axis=1)
                        mixed_segment = mix_audio_files(audio_segment, background_music)
                    combined_audio = np.concatenate((combined_audio, mixed_segment))
                    print('combined_audio:', combined_audio.shape)
                    print('当前情感：', most_common_emotion)
        # 保存最终合成的音频
        file_path = 'combined_audio.wav'
        sf.write(file_path, combined_audio, samplerate=rate)
    return file_path
