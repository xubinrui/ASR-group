'''
功能：进行语音降噪前后的对比绘图
负责人：徐彬芮
更新时间：2024.6.25

'''

import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt

# 音频路径
file_path ='model_prediction\data\demo.wav'

# 读取文件
data, rate = sf.read(file_path)

# 进行降噪
reduced_noise = nr.reduce_noise(y=data, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)



# 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Audio', color='blue')
plt.plot(reduced_noise, label='Noise-Reduced Audio', color='orange')
plt.title('Original vs Noise-Reduced Audio')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.show()