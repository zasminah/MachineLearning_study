import tensorflow as tf

x1 = tf.Variable(tf.constant(1.0))
x2 = tf.Variable(tf.constant(2.0))

# with tf.GradientTape() as tape: # GradientTape 라는 객체 생성, y=x1*x2
#     y = tf.multiply(x1,x2)
#
# gradients = tape.gradient(y,[x1,x2]) # y=x1*x2 에서 x1으로 미분할경우 x2로 미분할경우로 해서 [2.0,1.0]
# print(gradients[0].numpy())
# print(gradients[1].numpy())

x3 = tf.Variable(tf.constant(1.0))
a = tf.constant(2.0)

with tf.GradientTape() as tape:
    tape.watch(a) #상수미분에 필요, a를 변수처럼 쓸수 있게, 미분할 수 있게
    y=tf.multiply(x3,a)

gradient = tape.gradient(y,a) #a = constant
# watch가 있는 경우 : tf.Tensor(1.0, shape=(), dtype=float32)
# watch가 없는 경우 : None
print('\n', gradient)


