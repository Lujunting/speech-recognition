from sys import byteorder
from array import array
from struct import pack
import wave
import random
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# 音檔預處理
THRESHOLD = 500 #聲音閥值
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
# RATE = 44100  #取樣頻率
RATE = 48000  #取樣頻率(一秒鐘有多少個訊號。數值越高，音質越好。)

# 若小於閥值則停止錄音
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

# 音訊歸一化
def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r
    

# Trim the blank spots at the start and end
def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

# 將音訊頭尾加入一段無聲音的片段，以確保原音訊的重要資訊保留住
def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    return r

# 使用到is_silent 這個 function
def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')
    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb') # wb 為寫入檔案；rb 為讀取檔案
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


# 開始錄音 (0-99,100-199,200-299)=(cat,dog,wolf)
x0=20 
for i in range(0,x0):
    print('start to record')
    # x=random.randint(1,x0) # 隨機產生1~10數字來為音檔命名
    sound_file_path='C:/Users/User/Desktop/python_jupyter/py_file/sound_testdata/'
    record_to_file(rf"{sound_file_path}/demo{i}.wav")
    print('in the end')
結束錄音

#  畫音訊圖
audio_path = 'C:/Users/User/Desktop/python_jupyter/py_file/machine_learning_project/sound_data/bird/00f0204f_nohash_0.wav'
x , sr = librosa.load(audio_path)
plt.figure(figsize=(14, 6))
librosa.display.waveshow(x, sr=sr)
x = x/ np.max(x) 
print(np.min(x))
plt.show()
plt.xlabel('time')
plt.ylabel('amplitude')
