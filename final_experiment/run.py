'''
功能：运行主程序
负责人：徐彬芮
更新时间：2024.6.25

'''
import sys
import os
import time
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QMessageBox, QSlider, QProgressBar
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt, pyqtSignal, QObject
import soundfile as sf
import noisereduce as nr
import pyaudio
from classifier import classifier
import wave
from model_prediction.code.utils.record_audio import record_audio


class NewWindow(QWidget):
    def __init__(self, denoised_audio_path=''):
        super().__init__()
        self.denoised_audio_path = denoised_audio_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('新窗口 - 播放音频')
        
        # 分类器按钮
        self.classifierButton = QPushButton('运行音频预测与合成', self)
        self.classifierButton.clicked.connect(self.runClassifier)
        
        # 播放按钮和进度条
        self.playButton = QPushButton('播放音频', self)
        self.playButton.setEnabled(False)
        self.playButton.clicked.connect(self.playAudio)
        
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setEnabled(False)

        # 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.classifierButton)
        layout.addWidget(self.playButton)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        
        self.resize(400, 200)

        # 初始化播放器
        self.player = QMediaPlayer()
        self.player.durationChanged.connect(self.updateDuration)
        self.player.positionChanged.connect(self.updatePosition)

    def runClassifier(self):
        # 调用分类器并更新 denoised_audio_path
        self.denoised_audio_path = classifier(denoised_audio_path=self.denoised_audio_path)
        print(f"分类器处理后的音频文件：{self.denoised_audio_path}")
        # 启用播放按钮
        self.playButton.setEnabled(True)

    def playAudio(self):
        url = QUrl.fromLocalFile(self.denoised_audio_path)
        content = QMediaContent(url)
        self.player.setMedia(content)
        self.player.play()

    def updateDuration(self, duration):
        self.slider.setMaximum(duration)
        self.slider.setEnabled(True)

    def updatePosition(self, position):
        self.slider.setValue(position)

class RecordProgress(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

class RecordWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('录音中')
        self.layout = QVBoxLayout()
        self.label = QLabel('录音中...', self)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 60)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress)
        self.setLayout(self.layout)
        self.resize(300, 100)

    def update_progress(self, value):
        self.progress.setValue(value)

    def closeEvent(self, event):
        super().closeEvent(event)

class AudioOrTextApp(QWidget):
    def __init__(self):
        super().__init__()
        self.denoisedPath = ''
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Audio Processing')

        # 创建按钮
        self.audioButton = QPushButton('选择音频文件', self)
        self.label = QLabel('选择一种处理方式', self)
        
        # 播放按钮和进度条
        self.playOriginalButton = QPushButton('播放原音频', self)
        self.playDenoisedButton = QPushButton('播放去噪音频', self)
        self.originalSlider = QSlider(Qt.Horizontal, self)
        self.denoisedSlider = QSlider(Qt.Horizontal, self)

        self.playOriginalButton.setEnabled(False)
        self.playDenoisedButton.setEnabled(False)
        self.originalSlider.setEnabled(False)
        self.denoisedSlider.setEnabled(False)

        # 新的按钮
        self.newWindowButton = QPushButton('打开新窗口', self)
        self.recordButton = QPushButton('开始录音', self)

        # 连接按钮到处理函数
        self.audioButton.clicked.connect(self.processAudio)
        self.playOriginalButton.clicked.connect(lambda: self.playAudio(self.originalPath))
        self.playDenoisedButton.clicked.connect(lambda: self.playAudio(self.denoisedPath))
        self.newWindowButton.clicked.connect(self.openNewWindow)
        self.recordButton.clicked.connect(self.startRecording)

        # 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.audioButton)
        layout.addWidget(self.playOriginalButton)
        layout.addWidget(self.originalSlider)
        layout.addWidget(self.playDenoisedButton)
        layout.addWidget(self.denoisedSlider)
        layout.addWidget(self.newWindowButton)
        layout.addWidget(self.recordButton)
        self.setLayout(layout)

        self.resize(400, 400)

        # 初始化播放器
        self.player = QMediaPlayer()
        self.player.durationChanged.connect(self.updateDuration)
        self.player.positionChanged.connect(self.updatePosition)

    def processAudio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择音频文件', '', '音频文件 (*.wav)')
        if file_path:
            QMessageBox.information(self, '选择的文件', f'你选择的文件是: {file_path}')
            self.label.setText(f"选择的文件：{file_path}")
            self.originalPath = file_path
            self.handleAudioFile(file_path)

    def playAudio(self, file_path):
        if not file_path:
            return
        url = QUrl.fromLocalFile(file_path)
        content = QMediaContent(url)
        self.player.setMedia(content)
        self.player.play()

    def handleAudioFile(self, file_path):
        # 读取音频文件
        data, rate = sf.read(file_path)

        # 去噪处理
        # reduced_noise = nr.reduce_noise(y=data, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)
        reduced_noise=data
        # 保存去噪后的音频文件
        self.denoisedPath = os.path.splitext(file_path)[0] + '_denoised.wav'
        sf.write(self.denoisedPath, reduced_noise, rate)

        # 启用播放按钮和进度条
        self.playOriginalButton.setEnabled(True)
        self.playDenoisedButton.setEnabled(True)
        self.originalSlider.setEnabled(True)
        self.denoisedSlider.setEnabled(True)

    def updateDuration(self, duration):
        self.originalSlider.setMaximum(duration)
        self.denoisedSlider.setMaximum(duration)

    def updatePosition(self, position):
        if self.player.media().canonicalUrl().toLocalFile() == self.originalPath:
            self.originalSlider.setValue(position)
        elif self.player.media().canonicalUrl().toLocalFile() == self.denoisedPath:
            self.denoisedSlider.setValue(position)

    def openNewWindow(self):
        self.newWindow = NewWindow(self.denoisedPath)
        self.newWindow.show()
        self.close()

    def startRecording(self):
        self.recordWindow = RecordWindow()
        self.recordProgress = RecordProgress()
        self.recordProgress.progress.connect(self.recordWindow.update_progress)
        self.recordProgress.finished.connect(self.recordingFinished)
        self.recordWindow.show()

        # 创建两个线程：一个用于录音，一个用于更新进度条
        threading.Thread(target=self.recordAudioThread).start()
        threading.Thread(target=self.updateProgressBar).start()

    def recordAudioThread(self):
        record_audio(duration=60)  # 录音持续60秒
        self.recordProgress.finished.emit()

    def updateProgressBar(self):
        for i in range(60):
            time.sleep(1)
            self.recordProgress.progress.emit(i + 1)

    def recordingFinished(self):
        self.recordWindow.close()
        QMessageBox.information(self, '录音完成', '录音完成！')

def main():
    app = QApplication(sys.argv)
    mainWindow = AudioOrTextApp()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()