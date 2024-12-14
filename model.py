from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import numpy as np
import librosa
from features import augment_dataset

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

# 감정 예측 모델 정의
def build_emotion_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')  # 5가지 감정 (행복, 슬픔, 분노, 중립, 두려움)
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 딥보이스 예측을 위한 특징 추출 함수
def extract_features_for_deepvoice(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0)
    return features

# 감정 예측을 위한 특징 추출
def extract_emotion_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features = np.mean(chroma.T, axis=0)
    features = np.concatenate((features, np.mean(spectral_contrast.T, axis=0)))
    return features

# 모델 훈련 함수
def train_model(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (X_train.shape[1], 1)

    cnn_lstm_model = build_cnn_lstm_model(input_shape)
    cnn_lstm_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
    cnn_lstm_model.save('voice_model_cnn_lstm.h5')

# 모델 로드
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

