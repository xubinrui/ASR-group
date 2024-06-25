from pydub import AudioSegment

def stereo_to_mono(input_wav_path, output_wav_path):
    # 读取立体声音频文件
    stereo_audio = AudioSegment.from_wav(input_wav_path)
    
    # 将立体声转换为单声道
    mono_audio = stereo_audio.set_channels(1)
    
    # 保存单声道音频文件
    mono_audio.export(output_wav_path, format="wav")
    
    
# 示例用法
input_wav_path = r"D:\杂项\6月20日 (1)(1).WAV"
output_wav_path = "output_mono.wav"
stereo_to_mono(input_wav_path, output_wav_path)
