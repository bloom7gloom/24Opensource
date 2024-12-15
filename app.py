import tkinter as tk
from tkinter import filedialog
import numpy as np
import pyaudio
import librosa
import threading
from tensorflow.keras.models import load_model
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# 모델 로드
def load_trained_model():
    model_path = "voice_model_cnn_lstm.h5"  # 모델 파일 경로
    try:
        return load_model(model_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None

# 음성 특징 추출
def extract_features(audio_data, sr=22050):
    # librosa.load()는 파일을 읽지만, pyaudio로 받은 데이터는 바로 MFCC로 변환
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0)
    return np.pad(features, (0, 100 - len(features)), 'constant', constant_values=0)

# 마이크로부터 실시간으로 음성 녹음
def record_audio(duration=5, sr=22050):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sr,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    print("녹음을 시작합니다...")

    # 일정 시간 동안 녹음
    for i in range(0, int(sr / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("녹음이 종료되었습니다.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 바이너리 데이터를 numpy 배열로 변환
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
    return audio_data

# 신뢰도 계산 함수
def calculate_confidence(prob):
    if prob < 50:
        return "신뢰도: 낮음"
    elif 50 <= prob < 70:
        return "신뢰도: 보통"
    else:
        return "신뢰도: 높음"

# 실시간 예측을 위한 쓰레드
def real_time_predict(model, result_label):
    # 일정 시간 동안 음성을 녹음 (예: 5초)
    audio_data = record_audio(duration=5)

    # 음성 특징 추출
    features = extract_features(audio_data)
    features = np.expand_dims(features, axis=0)  # 2D 배열로 변환

    # 예측
    prediction = model.predict(features, verbose=0)[0][0]
    deepvoice_prob = prediction * 100  # 딥보이스 확률 계산
    normalvoice_prob = (1 - prediction) * 100  # 일반 목소리 확률 계산

    if prediction > 0.5:
        result = f"딥보이스일 확률: {deepvoice_prob:.2f}%"
    else:
        result = f"일반 목소리일 확률: {normalvoice_prob:.2f}%"

    # 신뢰도 추가
    confidence = calculate_confidence(deepvoice_prob if prediction > 0.5 else normalvoice_prob)

    result_label.config(text=f"{result}\n{confidence}")

# Tkinter GUI 클래스
class VoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("딥보이스 판별기")

        # 모델 로드
        self.model = load_trained_model()
        if self.model is None:
            self.root.quit()  # 모델이 없으면 애플리케이션 종료

        # UI 구성
        self.label = tk.Label(root, text="음성 파일을 업로드하거나 실시간 녹음을 시작하세요.", font=('Arial', 14))
        self.label.pack(pady=20)

        # 파일 업로드 버튼
        self.upload_button = tk.Button(root, text="파일 업로드", font=('Arial', 12), command=self.upload_file)
        self.upload_button.pack(pady=20)

        # 실시간 녹음 버튼
        self.record_button = tk.Button(root, text="실시간 녹음 시작", font=('Arial', 12), command=self.start_recording)
        self.record_button.pack(pady=20)

        # 결과 레이블
        self.result_label = tk.Label(root, text="", font=('Arial', 14))
        self.result_label.pack(pady=20)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.m4a;*.wav")])
        if file_path:
            audio_data, sr = librosa.load(file_path, sr=None)
            features = extract_features(audio_data, sr)
            features = np.expand_dims(features, axis=0)
            prediction = self.model.predict(features, verbose=0)[0][0]

            deepvoice_prob = prediction * 100
            normalvoice_prob = (1 - prediction) * 100

            if prediction > 0.5:
                result = f"딥보이스일 확률: {deepvoice_prob:.2f}%"
            else:
                result = f"일반 목소리일 확률: {normalvoice_prob:.2f}%"

            # 신뢰도 추가
            confidence = calculate_confidence(deepvoice_prob if prediction > 0.5 else normalvoice_prob)

            self.result_label.config(text=f"{result}\n{confidence}")

    def start_recording(self):
        # 실시간 녹음과 예측을 별도의 쓰레드에서 실행
        threading.Thread(target=real_time_predict, args=(self.model, self.result_label), daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceApp(root)
    root.mainloop()












