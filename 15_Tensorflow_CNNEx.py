from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#scale
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255 #4 tensor 형태 (Batch, Width, Height, Channel)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
#CNN part 32 : kernel 갯수, stride default = 1X1 ==> convolution 32개 나옴
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #kernel size 2X2, stride size도 동일하게 2X2
model.add(Dropout(0.25))#overfitting 줄이려고
#FC part
model.add(Flatten()) # data를 직렬화시켜줌  flatten (Flatten) (None, 9216)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) #0~9

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

result = model.fit(x_train, y_train,
                   validation_data=(x_test, y_test),
                   epochs=30,
                   batch_size=200,
                   verbose=2)

print('loss & accuracy:',model.evaluate(x_test, y_test))
y_vloss = result.history['val_loss']
y_loss = result.history['loss']
x_len = np.arange((len(y_loss)))
plt.plot(x_len, y_vloss, '.', c='red', label='validation loss')
plt.plot(x_len, y_loss, '.', c='blue', label='train loss')
plt.legend(loc='best')
plt.show()







