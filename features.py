from pydub import AudioSegment
import librosa
import numpy as np
import os
import random

# FFmpeg 경로 설정 (FFmpeg가 시스템에 설치되어 있어야 함)
AudioSegment.ffmpeg = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # 실제 FFmpeg 경로로 수정

# MP3 또는 M4A 파일에서 특징을 추출하는 함수
def extract_features(file_path):
    # 오디오 파일을 WAV로 변환 (MP3, M4A 모두 처리 가능)
    audio = AudioSegment.from_file(file_path)  # 이제 MP3와 M4A 모두 처리 가능
    wav_path = "temp.wav"  # 임시 WAV 파일 경로
    audio.export(wav_path, format="wav")

    # librosa를 사용하여 WAV 파일을 로드
    y, sr = librosa.load(wav_path, sr=None)

    # MFCC 추출
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # MFCC 평균값을 반환 (시간에 따른 평균)
    features = np.mean(mfcc.T, axis=0)

    # 특성 벡터 길이를 100으로 맞추기 위해 패딩 추가
    features_padded = np.pad(features, (0, 100 - len(features)), 'constant', constant_values=0)

    # 임시 파일 삭제
    os.remove(wav_path)

    return features_padded


def augment_audio(y, sr):
    augment_type = random.choice(["pitch", "speed", "noise", "none"])

    if augment_type == "pitch":
        # 피치 변경
        y = librosa.effects.pitch_shift(y, sr, n_steps=random.randint(-2, 2))

    elif augment_type == "speed":
        # 속도 변경
        speed_change = random.uniform(0.9, 1.1)  # 속도 변경 범위: 0.9배에서 1.1배 사이
        y = librosa.effects.time_stretch(y, speed_change)

    elif augment_type == "noise":
        # 노이즈 추가
        noise = np.random.randn(len(y)) * 0.005  # 작은 양의 가우시안 노이즈 추가
        y = y + noise
    elif augment_type == "volume":
        # 볼륨 변경
        volume_factor = random.uniform(0.7, 1.5)  # 볼륨을 0.7배에서 1.5배 사이로 랜덤 변경
        y = change_volume(y, volume_factor)

    return y

# 볼륨 변경 함수
def change_volume(y, factor=1.5):
    return y * factor
