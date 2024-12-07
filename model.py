import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam


# 딥러닝 모델 정의 (CNN + LSTM)
def build_model(input_shape):
    model = Sequential()

    # 1D CNN 레이어 (특징 추출)
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM 레이어 추가 (시간적 의존성 학습)
    model.add(LSTM(128, return_sequences=False))  # LSTM 추가, return_sequences=False는 출력이 1D로 나오도록 설정

    # 완전 연결층
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류

    # 모델 컴파일
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 딥러닝 모델 정의
def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 모델 훈련 함수
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    model.save('voice_model.h5')  # 훈련된 모델 저장

# 미리 훈련된 모델을 로드하는 함수
def load_trained_model():
    return load_model("C:/Users/puppy/PycharmProjects/test/voice_model.h5")


def extract_features(file_path):

    return np.random.random(100)  # 임시로 100개의 랜덤 숫자 리턴 (실제 구현 필요)

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
X = np.array([extract_features(f) for f in file_paths])  # 특징 추출 (X)
y = np.array(labels)  # 레이블 (y)

# 모델 훈련 함수 호출
train_model(X, y)
