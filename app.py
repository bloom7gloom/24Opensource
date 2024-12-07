import tkinter as tk
from tkinter import filedialog
from model import load_trained_model
from features import extract_features
import numpy as np
from tensorflow.keras.models import load_model

def load_trained_model():
    model_path = "C:\\Users\\puppy\\PycharmProjects\\test\\voice_model_cnn_lstm.h5"  # 모델 파일 경로
    return load_model(model_path)

class VoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("딥보이스 판별기")

        self.model = load_trained_model()
        self.label = tk.Label(root, text="음성 파일을 업로드해주세요.", font=('Arial', 14))
        self.label.pack(pady=20)

        self.upload_button = tk.Button(root, text="파일 업로드", font=('Arial', 12), command=self.upload_file)
        self.upload_button.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=('Arial', 14))
        self.result_label.pack(pady=20)

    def upload_file(self):
        # 파일 다이얼로그에서 MP3와 M4A 파일 모두 선택할 수 있도록 변경
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.m4a;*.wav")])
        if file_path:
            features = extract_features(file_path)
            features = np.expand_dims(features, axis=0)  # 2D 배열로 변환
            prediction = self.model.predict(features)
            if prediction > 0.5:
                result = "딥보이스입니다."
            else:
                result = "일반 목소리입니다."
            self.result_label.config(text=f"예측 결과: {result}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceApp(root)
    root.mainloop()
