import numpy as np
import numpy as np
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os

#录制音频
def record_audio(duration):
    import pyaudio
    import wave

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "./data/demo.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* 录音中...")

    frames = []

    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* 录音结束!")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

#语音分帧
def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i,:] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout

# 分割成語音段
def findSegment(express):
    
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg

# 使用能熵比进行语音分段
def pitch_vad(x, wnd, inc, T1, miniL=10):
    """
    :param x: 语音信号
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param T1: 端点检测阈值
    :param miniL: 语音段最小帧数
    """
    y = enframe(x, wnd, inc)    # 语音分帧
    fn = y.shape[0]
    if isinstance(wnd, int):    # 判断是wnd是窗函数还是窗长，并以此设定窗长
        wlen = wnd
    else:
        wlen = len(wnd)

    Sp = np.abs(np.fft.fft(y, axis=1))  # FFT变换
    Sp = Sp[:, :wlen // 2 + 1]  # 取正频率部分
    Esum = np.sum(np.multiply(Sp, Sp), axis=1)  # 能量总和
    prob = Sp / np.sum(Sp, axis=1, keepdims=True)   # 概率密度
    H = -np.sum(np.multiply(prob, np.log10(prob + 1e-16)), axis=1)  # 短时熵谱
    H = np.where(H < 0.1, np.max(H), H)
    Ef = np.sqrt(1 + np.abs(Esum / H))  # 能熵比
    Ef = Ef / np.max(Ef)    # 归一化

    zseg = findSegment(np.where(Ef > T1)[0])    # 找到能熵比大于门限的语音帧
    zsl = len(zseg.keys()) # 计算帧长
    SF = np.zeros(fn)
    # 标记有声段、噪声段
    for k in range(zsl):
        if zseg[k]['duration'] < miniL:
            zseg.pop(k)
        else:
            SF[zseg[k]['start']:zseg[k]['end']] = 1
    return zseg

#进行分段
def pitch_Corr(x, wnd, inc, T1, fs, miniL=100):
    """
    自相关法基音周期检测函数
    :param x: 语音信号
    :param wnd: 窗函数或窗长
    :param inc: 帧移,即分帧时每一帧跨过的样本数
    :param T1: 门限
    :param fs: 采样率
    :param miniL: 语音段的最小帧数
    :return voiceseg, vsl, SF, Ef, period: 语音段的起始点和终止点，语音段数，语音段标记，非语音段标记，基音周期
    """
    y = enframe(x, wnd, inc)  # 调用enframe方法进行分帧
    voiceseg = pitch_vad(x, wnd, inc, T1, miniL)  # 语音分段,它的作用是对语音信号进行活动检测，将语音段和非语音段进行区分。

    return voiceseg


#不画图版本
def co_voiceseg(wavefile,wlen = 6000,inc = 80,T1 = 0.05 ):

    '''
    wlen 分析窗口长度
    inc 连续窗口间隔
    time 时间向量
    T1 用于基音校正的阈值
    '''
    # 读取WAV文件
    (fs, data) = wavfile.read(wavefile)
    print(fs,len(data))

    # 去除直流偏移
    data = data - np.mean(data)

    # 幅值归一化
    data = data / np.max(data)

    # 分析参数
    N = len(data)

    # 进行基音校正并检测有声段、
    voiceseg = pitch_vad(data, wlen, inc, T1, miniL=100)

    return voiceseg

def save_segmented_audio(wavefile,voiceseg,inc=80, output_dir='./data/segments'):
    # 读取WAV文件
    fs, data = wavfile.read(wavefile)

    os.makedirs(output_dir, exist_ok=True)
    
    for idx, seg_info in voiceseg.items():
        start_frame = seg_info['start']
        end_frame = seg_info['end']
        
        # 提取语音段
        segment = data[start_frame*inc:end_frame*inc]
        
        # 构建输出文件名
        output_file = os.path.join(output_dir, f'segment_{idx}.wav')
        
        # 保存语音段为 WAV 文件
        wavfile.write(output_file, fs, segment)
        print(f'Saved segment {idx} to {output_file}')

# record_audio(duration=60) #录音5秒,并保存为demo.wav
wavefile='./data/demo.wav'
voiceseg=co_voiceseg(wavefile)
# 保存分段后的语音
save_segmented_audio(wavefile, voiceseg)
