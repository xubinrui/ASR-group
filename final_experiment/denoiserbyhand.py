import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft, wiener

def noise_estimation(noisy_stft, noise_frames=60):
    noise_est = np.mean(np.abs(noisy_stft[:, :noise_frames]) ** 2, axis=1)
    return noise_est

def spectral_subtraction(noisy_stft, noise_power_spectrum, alpha=20):
    noisy_power_spectrum = np.abs(noisy_stft) ** 2
    subtracted_spectrum = noisy_power_spectrum - alpha * noise_power_spectrum[:, None]
    subtracted_spectrum[subtracted_spectrum < 0] = 0
    return np.sqrt(subtracted_spectrum) * np.exp(1j * np.angle(noisy_stft))

def smooth_signal(signal, method='wiener', window_len=5):
    if method == 'wiener':
        return wiener(signal)
    elif method == 'moving_average':
        window = np.ones(window_len) / window_len
        return np.convolve(signal, window, mode='same')
    else:
        raise ValueError("Unknown smoothing method: {}".format(method))

def enhance_audio(input_wav, output_wav, noise_frames=60, alpha=20, smoothing_method='wiener'):
    # 读取音频文件
    rate, noisy_signal = wavfile.read(input_wav)

    # 参数设置
    n_fft = 1024
    hop_length = n_fft // 2
    win_length = n_fft

    # STFT
    f, t, Zxx = stft(noisy_signal, fs=rate, nperseg=win_length, noverlap=hop_length, nfft=n_fft)

    # 估计噪声功率谱
    noise_power_spectrum = noise_estimation(Zxx, noise_frames)

    # 进行谱减法
    enhanced_stft = spectral_subtraction(Zxx, noise_power_spectrum, alpha)

    # ISTFT
    _, enhanced_signal = istft(enhanced_stft, fs=rate, nperseg=win_length, noverlap=hop_length, nfft=n_fft)

    # 平滑处理
    enhanced_signal = smooth_signal(enhanced_signal, method=smoothing_method)

    # 写入输出文件
    wavfile.write(output_wav, rate, enhanced_signal.astype(np.int16))

    return rate, noisy_signal, enhanced_signal

# 使用示例
input_wav = r'D:\语音处理\ASR\ASR-group\final_experiment\中文朗诵生成\data\demo\demo.wav'
output_wav = 'enhanced_audio.wav'
rate, noisy_signal, enhanced_signal = enhance_audio(input_wav, output_wav)

# 绘制原始和增强音频的波形图
time_axis = np.arange(noisy_signal.shape[0]) / rate

plt.figure(figsize=(15, 6))

plt.subplot(2, 1, 1)
plt.plot(time_axis, noisy_signal, label='Noisy Signal')
plt.title('Noisy Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_axis, enhanced_signal, label='Enhanced Signal', color='orange')
plt.title('Enhanced Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
