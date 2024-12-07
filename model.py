from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split

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
    """
    X: 입력 데이터 (특징 벡터)
    y: 레이블
    CNN+LSTM 모델만 훈련
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (X_train.shape[1], 1)

    # CNN+LSTM 모델 훈련
    cnn_lstm_model = build_cnn_lstm_model(input_shape)
    cnn_lstm_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
    cnn_lstm_model.save('voice_model_cnn_lstm.h5')

# 훈련된 모델 로드 함수
def load_trained_model():
    """
    CNN+LSTM 모델 로드
    """
    model_path = 'voice_model_cnn_lstm.h5'
    try:
        return load_model(model_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None

# 특징 추출 함수 (임시 구현)
def extract_features(file_path):
    # 임시로 100개의 랜덤 숫자 리턴 (실제 구현 필요)
    return np.random.random(100)

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

# 음성 파일에서 특징 추출
X = np.array([extract_features(f) for f in file_paths])
y = np.array(labels)

# CNN+LSTM 모델 훈련
train_model(X, y)

