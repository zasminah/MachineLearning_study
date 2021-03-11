import tensorflow as tf
import numpy as np

# 1/1+e-(x1w1+x2w2+b)
x_train = np.array([[1., 1.],
                   [1., 2.],
                   [2., 1.],
                   [3., 2.],
                   [3., 3.],
                   [2., 3.]],
                   dtype=np.float32)

y_train = np.array([[0.],
                   [0.],
                   [0.],
                   [1.],
                   [1.],
                   [1.]],
                   dtype=np.float32)



tf.random.set_seed(12345)
W = tf.Variable(tf.random.normal([2, 1], mean=0.0)) #weight 2개 필요
b = tf.Variable(tf.random.normal([1], mean=0.0))

print('weights: \n', W.numpy(), '\n\nbias: \n', b.numpy())

learning_rate = 0.01


def predict(X):
    z = tf.matmul(X, W) + b
    hypothesis = 1 / (1 + tf.exp(-z))
    return hypothesis

for i in range(2001):
    with tf.GradientTape() as tape:
        hypothesis = predict(x_train)
        cost = tf.reduce_mean(-tf.reduce_sum(y_train*tf.math.log(hypothesis) + (1-y_train)*tf.math.log(1-hypothesis)))
        W_grad, b_grad = tape.gradient(cost, [W, b])#W와 b로 미분

        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)

    if i % 400 == 0:
        print("%s \n weights: \n%s \n bias: \n%s \n cost: %s\n" % (i, W.numpy(), b.numpy(), cost.numpy()))


hypo = predict(x_train)
print("hypothesis: \n", hypo.numpy())
print("result: \n", tf.cast(hypo > 0.5, dtype=tf.float32).numpy())
def acc(hypo, label):
    predicted = tf.cast(hypo > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, label), dtype=tf.float32))
    return accuracy

accuracy = acc(predict(x_train), y_train).numpy()
print("accuracy: %s" % accuracy)
