from sys import byteorder
from array import array
from struct import pack
import pyaudio
import wave
import random
import librosa
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


#  音檔特徵提取
# covert to mfcc vector
# Input: Folder Path
DATA_PATH = "C:/Users/User/Desktop/python_jupyter/py_file/machine_learning_project/sound_data/"

# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    print('label_indices:', label_indices)
    return labels, label_indices, to_categorical(label_indices)
# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    print(wave,sr)
    # cut audio file
    i=0
    # 訓練資料的長度
    wav_length=22050
    # # 音訊歸一化
    # wave = wave / np.max(wave) 
    # 聲音檔過長，擷取片段
    if len(wave) > wav_length:
        # 尋找最大聲的點，取前後各半
        i=np.argmax(wave)
        if i > (wav_length):
            wave = wave[i-int(wav_length/2):i+int(wav_length/2)]
        else:
            # 聲音檔過長，取前面
            wave = wave[0:wav_length]

    # 提取 mfcc特徵
    mfcc = librosa.feature.mfcc(wave, sr=2.5*sr)  # sr取樣頻率(16000 orign)

    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width < 0:
        pad_width = 0
        mfcc = mfcc[:,:11] 
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

        
def get_train_test(split_ratio=0.7, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays (將mfcc向量化為.npy檔形式)
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])
    
    # print(labels[0])
    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y) # 無太大用處，僅確認資料長度是否一致

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True) 
    # split_ratio作為訓練資料，1-split_ratio為測試資料