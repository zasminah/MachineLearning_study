import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
tf.random.set_seed(678)

import numpy as np

x_train = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y_train = np.array([0.,1.,1.,0.])

model = Sequential() #Model 객체생성
model.add(Dense(units=2,activation='sigmoid',input_dim=2))#Dense = layer 생성, units: 노드 수, activation func은 sigmoid 사용, input dim : 외부로부터 들어오는 input은 2개다
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy']) #어떠한 loss func을 사용할지, gridient descent 사용, metrics : 실행할때마다 loss와 accuracy를 찍어줘라

model.fit(x_train,y_train,epochs=50000) #실행시마다 미분시켜주는 것 : fit, 50000번 학습

print("first layer weights: ",model.layers[0].get_weights()[0])
print("first layer bias: ",model.layers[0].get_weights()[1])

print("second layer weights: ",model.layers[1].get_weights()[0])
print("second layer bias: ",model.layers[1].get_weights()[1])


print(model.predict(x_train)) #출력값 뽑아낼때 predict 사용
