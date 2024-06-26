'''
功能：将语音音频与背景音频混合
负责人：徐彬芮、崔彣婧
更新时间：2024.6.25

'''

import numpy as np
import wave
from pydub import AudioSegment
import noisereduce as nr

# 读取WAV文件
def read_wave_file(file_path):
    with wave.open(file_path, 'rb') as wf:
        params = wf.getparams()
        frames = wf.readframes(params.nframes)
        audio = np.frombuffer(frames, dtype=np.int16)
    return params, audio

# 写入WAV文件
def write_wave_file(file_path, params, audio):
    with wave.open(file_path, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(audio.tobytes())

# 转换MP3文件为WAV文件
def convert_mp3_to_wav(mp3_file):
    wav_file = "background_temp.wav"
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")
    return wav_file

# 混合音频文件
def mix_audio_files(speech_audio, background_audio, background_volume=0.4):
    # 调整背景音频的音量
    background_audio = background_audio * background_volume
    print (speech_audio.shape, background_audio.shape)
    # 确保两个音频文件具有相同的长度
    if len(speech_audio) > len(background_audio):
        background_audio = np.pad(background_audio, (0, len(speech_audio) - len(background_audio)), 'wrap')
    else:
        background_audio = background_audio[:len(speech_audio)]

    # 混合音频
    mixed_audio = speech_audio + background_audio
    mixed_audio = np.clip(mixed_audio, -32768, 32767)  

    #  # 将混合后的音频写入输出文件
    output_file = "combined_audio.wav"
    # write_wave_file(output_file, speech_params, mixed_audio)

    return mixed_audio