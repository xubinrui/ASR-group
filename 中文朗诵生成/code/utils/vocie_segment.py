'''
功能：
1.分割语音

最后更新时间：
2024.6.15

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

#画图板
def pitch_vad(x, wnd, inc, T1, miniL=10):
    """
    使用能熵比检测基音，实际上就是语音分段
    :param x: 语音信号
    :param wnd: 窗函数或窗长
    :param inc: 帧移
    :param T1: 端点检测阈值
    :param miniL: 语音段最小帧数
    :return:
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
    return zseg, len(zseg.keys()), SF, Ef

# #不画图版本
# def pitch_vad(x, wnd, inc, T1, miniL=10):
#     """
#     使用能熵比检测基音，实际上就是语音分段
#     :param x: 语音信号
#     :param wnd: 窗函数或窗长
#     :param inc: 帧移
#     :param T1: 端点检测阈值
#     :param miniL: 语音段最小帧数
#     :return:
#     """
#     y = enframe(x, wnd, inc)    # 语音分帧
#     fn = y.shape[0]
#     if isinstance(wnd, int):    # 判断是wnd是窗函数还是窗长，并以此设定窗长
#         wlen = wnd
#     else:
#         wlen = len(wnd)

#     Sp = np.abs(np.fft.fft(y, axis=1))  # FFT变换
#     Sp = Sp[:, :wlen // 2 + 1]  # 取正频率部分
#     Esum = np.sum(np.multiply(Sp, Sp), axis=1)  # 能量总和
#     prob = Sp / np.sum(Sp, axis=1, keepdims=True)   # 概率密度
#     H = -np.sum(np.multiply(prob, np.log10(prob + 1e-16)), axis=1)  # 短时熵谱
#     H = np.where(H < 0.1, np.max(H), H)
#     Ef = np.sqrt(1 + np.abs(Esum / H))  # 能熵比
#     Ef = Ef / np.max(Ef)    # 归一化

#     zseg = findSegment(np.where(Ef > T1)[0])    # 找到能熵比大于门限的语音帧
#     zsl = len(zseg.keys()) # 计算帧长
#     SF = np.zeros(fn)
#     # 标记有声段、噪声段
#     for k in range(zsl):
#         if zseg[k]['duration'] < miniL:
#             zseg.pop(k)
#         else:
#             SF[zseg[k]['start']:zseg[k]['end']] = 1
#     return zseg

def FrameTimeC(frameNum, frameLen, inc, fs):
    ll = np.array([i for i in range(frameNum)])
    return ((ll - 1) * inc + frameLen / 2) / fs

#绘图板
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
    fn = y.shape[0] #存储分帧后的帧数
    voiceseg, vsl, SF, Ef = pitch_vad(x, wnd, inc, T1, miniL)  # 语音分段,它的作用是对语音信号进行活动检测，将语音段和非语音段进行区分。
    lmin = fs // 500  # 基音周期的最小值
    lmax = fs // 200  # 基音周期的最大值
    period = np.zeros(fn)
    for i in voiceseg.keys():
           # 在所有语音段中
        ixb = voiceseg[i]['start']  # 语音段的起始帧
        ixd = voiceseg[i]['duration']  # 语音段的帧数
        for k in range(ixd):
            # TODO 需要补充：输入y, 调用np.correlate计算短时自相关,并找到最大值,返回自相关函数ru
            frame = y[ixb + k]  # 当前帧
            ru = np.correlate(frame, frame, mode='full')  #np.correlate 计算信号与其自身的滞后版本的相关性。模式'full'返回每一个可能延迟的自相关值
            ru = ru[len(frame)-1:]  # 保留正半部分，因为自相关函数是对称的
            # TODO 需要补充：输入ru找到范围内短时自相关最大值的标号,将其作为基音周期的估值,并存入period
            rmaxIndex = np.argmax(ru[lmin:lmax+1]) + lmin  #  ru[lmin:lmax+1] 是指在允许的延时范围内搜索自相关最大值的索引，然后加上 lmin 得到实际的样本延迟
            period[ixb+k] = rmaxIndex  # 基音周期估值
    return voiceseg, vsl, SF, Ef, period

# #不会图版
# def pitch_Corr(x, wnd, inc, T1, fs, miniL=100):
#     """
#     自相关法基音周期检测函数
#     :param x: 语音信号
#     :param wnd: 窗函数或窗长
#     :param inc: 帧移,即分帧时每一帧跨过的样本数
#     :param T1: 门限
#     :param fs: 采样率
#     :param miniL: 语音段的最小帧数
#     :return voiceseg, vsl, SF, Ef, period: 语音段的起始点和终止点，语音段数，语音段标记，非语音段标记，基音周期
#     """
#     y = enframe(x, wnd, inc)  # 调用enframe方法进行分帧
#     fn = y.shape[0] #存储分帧后的帧数
#     voiceseg = pitch_vad(x, wnd, inc, T1, miniL)  # 语音分段,它的作用是对语音信号进行活动检测，将语音段和非语音段进行区分。
#     lmin = fs // 500  # 基音周期的最小值
#     lmax = fs // 200  # 基音周期的最大值
#     period = np.zeros(fn)
#     for i in voiceseg.keys():
#            # 在所有语音段中
#         ixb = voiceseg[i]['start']  # 语音段的起始帧
#         ixd = voiceseg[i]['duration']  # 语音段的帧数
#         for k in range(ixd):
#             # TODO 需要补充：输入y, 调用np.correlate计算短时自相关,并找到最大值,返回自相关函数ru
#             frame = y[ixb + k]  # 当前帧
#             ru = np.correlate(frame, frame, mode='full')  #np.correlate 计算信号与其自身的滞后版本的相关性。模式'full'返回每一个可能延迟的自相关值
#             ru = ru[len(frame)-1:]  # 保留正半部分，因为自相关函数是对称的
#             # TODO 需要补充：输入ru找到范围内短时自相关最大值的标号,将其作为基音周期的估值,并存入period
#             rmaxIndex = np.argmax(ru[lmin:lmax+1]) + lmin  #  ru[lmin:lmax+1] 是指在允许的延时范围内搜索自相关最大值的索引，然后加上 lmin 得到实际的样本延迟
#             period[ixb+k] = rmaxIndex  # 基音周期估值
#     return voiceseg

#绘图板
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
    time = [i / fs for i in range(N)]  # 时间向量

    # 进行基音校正并检测有声段
    voiceseg, _, SF, _, period = pitch_Corr(data, wlen, inc, T1, fs)

    # 计算帧数
    fn = len(SF)
    print('fn:',fn)

    # 计算帧时间位置
    frameTime = FrameTimeC(fn, wlen, inc, fs)

    # 设置子图布局
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.5)  # 调整垂直间距

    # 绘制波形图
    axs[0].plot(time, data)
    axs[0].set_title('Waveform of speech')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlabel('Time/s')

    # 绘制自相关基音周期检测图
    axs[1].plot(frameTime, period)
    axs[1].set_title('Pitch Detection by Autocorrelation')
    axs[1].set_ylabel('Period')
    axs[1].set_xlabel('Time/s')

    # 标记有声段
    for i in voiceseg.keys():
        nx1 = voiceseg[i]['start']
        nx2 = voiceseg[i]['end']

        # 在波形图上标记有声段
        axs[0].axvline(frameTime[nx1], np.min(data), np.max(data), color='blue', linestyle='--')
        axs[0].axvline(frameTime[nx2], np.min(data), np.max(data), color='red', linestyle='-')

        #在自相关基音周期检测图上标记有声段
        axs[1].axvline(frameTime[nx1], np.min(period), np.max(period), color='blue', linestyle='--')
        axs[1].axvline(frameTime[nx2], np.min(period), np.max(period), color='red', linestyle='-')

    axs[0].legend(['Waveform', 'Start', 'End'])
    axs[1].legend(['Pitch', 'Start', 'End'])

    os.makedirs('figs', exist_ok=True) # 创建文件夹
    plt.savefig('figs/pitch.png') # 保存图片
    plt.show()
    return voiceseg

# #不画图版本
# def co_voiceseg(wavefile,wlen = 6000,inc = 80,T1 = 0.05 ):

#     '''
#     wlen 分析窗口长度
#     inc 连续窗口间隔
#     time 时间向量
#     T1 用于基音校正的阈值
#     '''
#     # 读取WAV文件
#     (fs, data) = wavfile.read(wavefile)
#     print(fs,len(data))

#     # 去除直流偏移
#     data = data - np.mean(data)

#     # 幅值归一化
#     data = data / np.max(data)

#     # 分析参数
#     N = len(data)
#     time = [i / fs for i in range(N)]  # 时间向量

#     # 进行基音校正并检测有声段
#     voiceseg = pitch_Corr(data, wlen, inc, T1, fs)

#     return voiceseg

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

if __name__ == "__main__":
    wavefile='././data/demo.wav'
    voiceseg=co_voiceseg(wavefile)
    # 保存分段后的语音
    save_segmented_audio(wavefile, voiceseg)
    pass
