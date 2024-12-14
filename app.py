import tkinter as tk
from tkinter import filedialog
from model import load_trained_model
from features import extract_features
import numpy as np
from tensorflow.keras.models import load_model

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
            prediction = self.model.predict(features)[0][0]  # 예측값 추출

            deepvoice_prob = prediction * 100  # 딥보이스 확률 계산
            normalvoice_prob = (1 - prediction) * 100  # 일반 목소리 확률 계산

            # 예측 결과와 신뢰도 계산
            if prediction > 0.5:
                confidence = '높음' if deepvoice_prob > 70 else '보통' if deepvoice_prob > 50 else '낮음'
                result = f"예측 결과: 딥보이스일 확률: {deepvoice_prob:.2f}%\n신뢰도: {confidence}"
            else:
                confidence = '높음' if normalvoice_prob > 70 else '보통' if normalvoice_prob > 50 else '낮음'
                result = f"예측 결과: 일반 목소리일 확률: {normalvoice_prob:.2f}%\n신뢰도: {confidence}"

            # 결과 출력
            self.result_label.config(text=result)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceApp(root)
    root.mainloop()



