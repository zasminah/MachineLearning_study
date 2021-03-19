from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import numpy
import pandas as pd
import tensorflow as tf

numpy.random.seed(777)
tf.random.set_seed(777)

df = pd.read_csv('13_sonar.csv', header=None)

dataset = df.values
x_data = dataset[:,0:60].astype(float)
y_data = dataset[:,60]

e = LabelEncoder()
e.fit(y_data)
y_data = e.transform(y_data) #R,M => 0,1로 변경

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=48) #k-fold, 10개로 나누겠다.

accuracy = []

for train, test in skf.split(x_data, y_data):
    model = Sequential()
    model.add(Dense(30, input_dim=60, activation='relu')) #첫번째 히든 층 30개 node
    model.add(Dense(10, activation='relu')) #두번째 히든 10개
    model.add(Dense(1, activation='sigmoid')) #출력은 1개
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_data[train], y_data[train], epochs=100, batch_size=5)
    k_accuracy = "%.3f" % (model.evaluate(x_data[test], y_data[test])[1])
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy:" % n_fold, accuracy)
