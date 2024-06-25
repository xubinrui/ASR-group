import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt

# Define the file path (example path, should be replaced with actual file path)
file_path = r'D:\语音处理\ASR\ASR-group\final_experiment\中文朗诵生成\data\demo\demo.wav'

# Read audio file
data, rate = sf.read(file_path)

# Apply noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)



# Plotting the original and noise-reduced audio together
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Audio', color='blue')
plt.plot(reduced_noise, label='Noise-Reduced Audio', color='orange')
plt.title('Original vs Noise-Reduced Audio')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.show()