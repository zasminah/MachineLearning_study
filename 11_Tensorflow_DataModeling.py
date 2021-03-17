from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np #배열형태로 읽어올 수 있음 일반적으로는 pandas 많이 사용
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

data = np.loadtxt('11_ThoraricSurgery.csv', delimiter=',') #각행의 구분자는 , : csv file이기때문에.. 배열형태로 가져옴

x_data = data[:, 0:17]
y_data = data[:, 17]

model = Sequential() #모델객체
model.add(Dense(30, input_dim=17, activation='relu')) #첫번째 hidden
model.add(Dense(12, activation='relu')) #Hidden layer
model.add(Dense(1, activation='sigmoid')) #출력층

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary() #모델의 형태가 어떻게 구성되었는지 print

#학습을 위해 x/y data, ephohs = 30, 30번 학습하겠다 전체 data를 몇번 사용하겠느냐.
# batch_size :  data를 10개씩 자르는것 그래서 한번에 10개의 data를 한번에 넣어주고 평균을 내고 학습.
# 하나씩 하나씩 넣는 것 보다 좀더 안정적인 학습을 할 수 있도록 도와줌 => 학습의 속도를 높여줌
model.fit(x_data, y_data, epochs=30, batch_size=10)