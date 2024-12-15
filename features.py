import os
import numpy as np
import librosa
import random
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


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
def augment_dataset(a_folder, b_folder):
    augmented_X, augmented_y = [], []

    # 딥보이스 (a_folder)와 일반 목소리 (b_folder)에서 파일을 자동으로 로드
    a_files = [os.path.join(a_folder, f) for f in os.listdir(a_folder) if f.endswith(('.mp3', '.wav', '.m4a'))]
    b_files = [os.path.join(b_folder, f) for f in os.listdir(b_folder) if f.endswith(('.mp3', '.wav', '.m4a'))]

    # 딥보이스 데이터를 먼저 추가
    for file_path in a_files:
        augmented_X.append(extract_features(file_path))
        augmented_y.append(1)  # 1: 딥보이스

        # 증강 데이터 생성
        for _ in range(3):  # 3개의 증강 데이터 생성
            augmented_X.append(extract_features_with_augmentation(file_path, augment=True))
            augmented_y.append(1)

    # 일반 목소리 데이터를 추가
    for file_path in b_files:
        augmented_X.append(extract_features(file_path))
        augmented_y.append(0)  # 0: 일반 목소리

        # 증강 데이터 생성
        for _ in range(3):  # 3개의 증강 데이터 생성
            augmented_X.append(extract_features_with_augmentation(file_path, augment=True))
            augmented_y.append(0)

    return np.array(augmented_X), np.array(augmented_y)


def augment_dataset_from_folders(a_folder, b_folder):
    a_files = [os.path.join(a_folder, f) for f in os.listdir(a_folder) if f.endswith(('.mp3', '.wav', '.m4a'))]
    b_files = [os.path.join(b_folder, f) for f in os.listdir(b_folder) if f.endswith(('.mp3', '.wav', '.m4a'))]

    # a_folder에서 딥보이스 데이터 처리
    augmented_X, augmented_y = [], []

    for file_path in a_files:
        augmented_X.append(extract_features_with_augmentation(file_path, augment=True))
        augmented_y.append(1)  # 딥보이스 레이블

    # b_folder에서 일반 목소리 데이터 처리
    for file_path in b_files:
        augmented_X.append(extract_features_with_augmentation(file_path, augment=True))
        augmented_y.append(0)  # 일반 목소리 레이블

    return np.array(augmented_X), np.array(augmented_y)
