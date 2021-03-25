import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN
from tensorflow.keras.models import Model
import numpy as np

tf.random.set_seed(1)
np.random.seed(1)
inputs = Input(shape=(1,2)) #Input (batch, timestep, imput dimension) ==> batch 제외, timestep =1 imput dim = 2
output, state = SimpleRNN(3, return_state=True)(inputs) #RNN 출력 [o1, o2, o3] hidden size
model = Model(inputs=inputs, outputs=[output, state])

data = np.array([[ [1,2] ]]) #3차원,
output, state = model.predict(data) #예측치
print("output: ",output)
print("state: ",state)

# I      [1,0,0,0]
# work   [0,1,0,0]
# at     [0,0,1,0]
# home [0,0,0,1]
#
# I work at home =  [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ]
# I home at work =  [ [1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0] ]

data = np.array([
    [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ],
    [ [1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0] ]
])

inputs = Input(shape=(4, 4))

output, state = SimpleRNN(3, return_state=True, return_sequences=True)(inputs)
model = Model(inputs=inputs, outputs=[output, state])

output, state = model.predict(data)
print('\n',output)