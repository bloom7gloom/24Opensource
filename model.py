from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import numpy as np
import librosa
from features import augment_dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


# 딥보이스 예측을 위한 CNN + LSTM 모델 정의
def build_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # 딥보이스 vs 일반 목소리
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 모델 훈련 함수
def train_model(a_folder, b_folder):
    # 데이터 증강 및 확장
    X, y = augment_dataset(a_folder, b_folder)

    # 데이터 분할
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], 1)  # 입력 형태 (특징 크기, 1채널)

    cnn_lstm_model = build_cnn_lstm_model(input_shape)

    # 훈련
    cnn_lstm_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=32)
    cnn_lstm_model.save('voice_model_cnn_lstm.h5')

# 모델 로드
def load_trained_model():
    model_path = 'voice_model_cnn_lstm.h5'
    try:
        return load_model(model_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None

# 훈련 데이터 준비 (폴더 경로 수정)
a_folder = "C:\\Users\\puppy\\ex\\deep"  # 딥보이스 데이터가 저장된 폴더 경로
b_folder = "C:\\Users\\puppy\\ex\\real"  # 일반 목소리 데이터가 저장된 폴더 경로

# 모델 훈련
train_model(a_folder, b_folder)



