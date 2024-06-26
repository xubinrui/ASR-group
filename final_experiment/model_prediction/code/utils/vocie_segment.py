'''
功能：
1.分割语音

最后更新时间：
2024.6.20

负责人：崔彣婧
'''

import numpy as np
import numpy as np
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os

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

# 分割成语音段
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

# 使用能熵比检测有声段
def pitch_vad(x, wnd, inc, T1, miniL=10):

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

#计算所有语音的有声段并进行分段
def co_voiceseg(wavefile,wlen = 6000,inc = 80,T1 = 0.05 ):

    # 读取WAV文件
    (fs, data) = wavfile.read(wavefile)
    print(fs,len(data))

    # 去除直流偏移
    data = data - np.mean(data)

    # 幅值归一化
    data = data / np.max(data)

    # 检测有声段
    voiceseg = pitch_vad(data, wlen, inc, T1, miniL=100)

    return voiceseg

# 将所有分段后的有声段下载到segments文件夹
def save_segmented_audio(wavefile,voiceseg,inc=80, output_dir='./segments'):
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


if __name__ == "__main__":
    # # record_audio(duration=60) #录音5秒,并保存为demo.wav
    # wavefile='./data/demo.wav'
    # voiceseg=co_voiceseg(wavefile)
    # # 保存分段后的语音
    # save_segmented_audio(wavefile, voiceseg)
    pass
