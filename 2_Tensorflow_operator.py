import tensorflow as tf

a = tf.range(6, dtype=tf.int32)
b = tf.ones(6, dtype=tf.int32)*2
# print(a.numpy())
# print(b.numpy())
print(tf.add(a,b).numpy())
print((a+b).numpy())
print((-b).numpy())

print('\n',tf.maximum(a,b).numpy()) #Compare the value in each index and return the maximum value

print('\n', tf.reduce_sum(a).numpy()) #Add all of a array as scalar type

print('\n', tf.reduce_mean(a).numpy()) #Mean value

c = tf.constant([[2,5,3],[4,6,8]])
# print('\n', c)
print('\n', tf.reduce_mean(c, axis=1).numpy()) #axis=1 row ,axis=0 column

d = tf.constant([[2,0],[0,1]], dtype=tf.float32)
e = tf.constant([[1,1],[1,1]], dtype=tf.float32)
print('dXe:', tf.matmul(d,e).numpy())
print('d-1:', tf.linalg.inv(d).numpy())