from 語音辨識_prepartion_20220516 import *
# import photo_recognition_prediction_20220428 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json


# sound_jason file
with open('語音辨識_training_20220516.json','r') as f:
    model_json_sound = json.load(f)
loaded_model_sound = model_from_json(model_json_sound)
loaded_model_sound .load_weights('語音辨識_training_20220516.h5')


# import photo_recognition_prediction_20220428 result 
# print('start to record')
# sound_file_path='C:/Users/User/Desktop/python_jupyter/py_file/'
# record_to_file(rf"{sound_file_path}/demo_dog.wav")
# print('in the end') 

clt=['bed','bird','cat','dog','down']
# test for sound file 
mfcc = wav2mfcc('C:/Users/User/Desktop/python_jupyter/py_file/dog.wav')
mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)  #20個11*1的列向量
print("labels=", get_labels())
speech_result= np.argmax(loaded_model_sound.predict(mfcc_reshaped))
print("predict=",clt[speech_result])
# photo_recognition_prediction_20220428.sound_to_photo(speech_result)





