import numpy as np
import librosa
import random
from sklearn.model_selection import train_test_split

# 특징 추출 함수 (MFCC)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13개의 MFCC 계수 추출
    features = np.mean(mfcc.T, axis=0)  # MFCC의 평균을 추출
    return np.pad(features, (0, 100 - len(features)), 'constant', constant_values=0)  # 길이를 맞추기 위해 패딩

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

# 데이터 증강을 포함한 특징 추출
def extract_features_with_augmentation(file_path, augment=False):
    y, sr = librosa.load(file_path, sr=None)

    if augment:
        y = augment_audio(y, sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13개의 MFCC 계수 추출
    features = np.mean(mfcc.T, axis=0)  # MFCC의 평균을 추출
    return np.pad(features, (0, 100 - len(features)), 'constant', constant_values=0)  # 길이를 맞추기 위해 패딩

# 데이터 증강 후 데이터셋 확장
def augment_dataset(file_paths, labels):
    augmented_X, augmented_y = [], []

    for file_path, label in zip(file_paths, labels):
        # 원본 데이터
        augmented_X.append(extract_features(file_path))
        augmented_y.append(label)

        # 증강 데이터 생성
        for _ in range(3):  # 3개의 증강 데이터 생성
            augmented_X.append(extract_features_with_augmentation(file_path, augment=True))
            augmented_y.append(label)

    return np.array(augmented_X), np.array(augmented_y)

# 파일 경로 및 레이블
file_paths = [
    "C:\\Users\\puppy\\ex\\1.1.mp3",
    "C:\\Users\\puppy\\ex\\1.2.mp3",
    "C:\\Users\\puppy\\ex\\1.3.mp3",
    "C:\\Users\\puppy\\ex\\0.1.mp3",
    "C:\\Users\\puppy\\ex\\0.2.mp3",
    "C:\\Users\\puppy\\ex\\0.3.mp3"
]
labels = [1, 1, 1, 0, 0, 0]  # 1: 딥보이스, 0: 일반 목소리

# 데이터 증강 및 확장
X, y = augment_dataset(file_paths, labels)

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"훈련 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_val.shape}")
