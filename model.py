from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import random

# 데이터 증강 함수
def augment_audio(y, sr):
    augment_type = random.choice(["pitch", "speed", "noise", "none"])

    if augment_type == "pitch":
        y = librosa.effects.pitch_shift(y, sr, n_steps=random.randint(-2, 2))  # 피치 변경
    elif augment_type == "speed":
        speed_change = random.uniform(0.9, 1.1)  # 속도 변경
        y = librosa.effects.time_stretch(y, speed_change)
    elif augment_type == "noise":
        noise = np.random.randn(len(y)) * 0.005  # 작은 노이즈 추가
        y = y + noise

    return y

# 특징 추출 함수 (원본 데이터)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0)
    return np.pad(features, (0, 100 - len(features)), 'constant', constant_values=0)

# 증강 포함 특징 추출
def extract_features_with_augmentation(file_path, augment=False):
    y, sr = librosa.load(file_path, sr=None)

    if augment:
        y = augment_audio(y, sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0)
    return np.pad(features, (0, 100 - len(features)), 'constant', constant_values=0)

# 데이터 증강 및 데이터셋 확장
def augment_dataset(file_paths, labels):
    augmented_X, augmented_y = [], []

    for file_path, label in zip(file_paths, labels):
        # 원본 데이터
        augmented_X.append(extract_features(file_path))
        augmented_y.append(label)

        # 증강 데이터 추가
        for _ in range(3):  # 각 데이터에 대해 3개의 증강 데이터 생성
            augmented_X.append(extract_features_with_augmentation(file_path, augment=True))
            augmented_y.append(label)

    return np.array(augmented_X), np.array(augmented_y)

# CNN + LSTM 모델 정의
def build_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 모델 훈련 함수
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (X_train.shape[1], 1)

    cnn_lstm_model = build_cnn_lstm_model(input_shape)
    cnn_lstm_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
    cnn_lstm_model.save('voice_model_cnn_lstm.h5')

# 훈련된 모델 로드 함수
def load_trained_model():
    model_path = 'voice_model_cnn_lstm.h5'
    try:
        return load_model(model_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None

# 훈련 데이터 준비 (파일 경로 수정)
file_paths = [
    "C:\\Users\\puppy\\ex\\1.1.mp3",
    "C:\\Users\\puppy\\ex\\1.2.mp3",
    "C:\\Users\\puppy\\ex\\1.3.mp3",
    "C:\\Users\\puppy\\ex\\0.1.mp3",
    "C:\\Users\\puppy\\ex\\0.2.mp3",
    "C:\\Users\\puppy\\ex\\0.3.mp3"
]
labels = [1, 1, 1, 0, 0, 0]  # 1: 딥보이스, 0: 일반 목소리

# 음성 파일에서 특징 추출 및 데이터 증강
X, y = augment_dataset(file_paths, labels)

# 모델 훈련
train_model(X, y)
