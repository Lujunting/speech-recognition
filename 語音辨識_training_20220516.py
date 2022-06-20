from 語音辨識_prepartion_20220516 import *
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam,RMSprop
import json
import time # 將訓練時間記錄下來
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from keras.layers import Input, Dense, SimpleRNN, RNN, LSTM

# 導入tensorboard追蹤訓練趨勢 ；
NAME = f'speech_recog_{int(time.time())}'  # time 會以數字編碼呈現
tensorboard = TensorBoard(log_dir=f'sound_data/{NAME}') # save as .log file

# Save data to array file first
save_data_to_array(path=DATA_PATH, max_pad_len=11)

# 載入 data 資料夾的訓練資料，並自動分為『訓練組』及『測試組』
X_train, X_test, y_train, y_test = get_train_test()  # import get_train_test fn
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

# 類別變數轉為one-hot encoding，適用在無序型的類別型資料，將這些特徵轉換成二元特徵，意即每個類別都是一個新的特徵，是這個類別就給1，不是就給0
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
print("X_train.shape=", X_train.shape)

# 建立簡單的線性執行的模型
model = Sequential()
# 建立卷積層，Kernal Size: 2x2, activation function 採用 relu
model.add(Conv2D(64, (3,3), activation='relu',input_shape=(20, 11, 1)))
# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# 全連接層: 128個output
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
# Add output layer
model.add(Dense(5, activation='softmax'))
# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss=keras.losses.categorical_crossentropy,
            # optimizer=keras.optimizers.Adadelta(),
              optimizer=Adam(learning_rate=0.00001),
              metrics=['accuracy'])

model.summary()
callback=[EarlyStopping(monitor='val_acc',patience=3,mode='max')] # 發現acc有下降趨勢提早停止
history=model.fit(X_train, y_train_hot, batch_size=30, epochs=250, verbose=2, validation_data=(X_test, y_test_hot),callbacks=callback)

# save as jason 
model_json = model.to_json()
with open("語音辨識_training_20220620.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save("語音辨識_training_20220620.h5")


# plot training loss and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
#繪製訓練 & 驗證準確值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

