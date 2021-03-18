from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd #문자열이 있어서 numpy로 배열형태로 읽을 수 없으므로 pandas 사용
import numpy
import tensorflow as tf

numpy.random.seed(777)
tf.random.set_seed(777)

df = pd.read_csv('12_iris.csv')
dataset = df.values #data만 추출할때 values
x_data = dataset[:,0:4].astype(float) #x data 4개
y_data = dataset[:,4] # y data 마지막값

#y data를 index로 변경해야함
e = LabelEncoder() # y data를 받아서 unique한 값만 뽑아냄, 순서대로 index를 가지고 있음
e.fit(y_data)
y_data = e.transform(y_data)#변환시키는 부분, 0,1,2로 변환됨
#one-hot code
y_encoded = to_categorical(y_data)# 이 data는 종류가 총 3가지로 [0,0,1] [0,1,0],[0,0,1]로 변환필요, one-hot-encoder


model = Sequential()
model.add(Dense(20,  input_dim=4, activation='relu'))#입력 총 4개,
model.add(Dense(3, activation='softmax'))#출력 layer

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

model.fit(x_data, y_encoded, epochs=50, batch_size=5)

result = model.evaluate(x_data, y_encoded) #평가
print("\n loss: %.4f" % (result[0]))
print("\n Accuracy: %.4f" % (result[1]))