from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import  to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


np.random.seed(3)
tf.random.set_seed(3)

(x_train, y_train), (x_test, y_test) = mnist.load_data() #각각 60000개와 10000개로 구성
x_train = x_train.reshape(x_train.shape[0], 784).astype(float)/255 #scale값 조정

y_train = to_categorical(y_train, 10) #명확하게 하기 위해 가짓수가 10가지라고 명시
y_test = to_categorical(y_test, 10)


model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''
verbose 0: silence
verbose 1: progressbar
verbose 2: oneline acc, loss
'''
result = model.fit(x_train, y_train,
                   validation_split=0.3, #data가 들어왔을 경우 train/test => train은 train/validate로 나눌 수 있음 - overfitting
                   epochs=20,
                   batch_size=200,
                   verbose=2) #print mode
print('accuracy:', model.evaluate(x_test, y_test)[1])
y_vloss = result.history['val_loss']
y_loss = result.history['loss']
x_len = np.arange((len(y_loss)))
plt.plot(x_len, y_vloss, '.', c='red', label='validation loss')
plt.plot(x_len, y_loss, '.', c='blue', label='train loss')
plt.legend(loc='best')
plt.show()








