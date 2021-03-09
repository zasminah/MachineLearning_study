import tensorflow as tf

a = tf.constant([1,2,3], dtype=tf.float32)
print('a:', a.numpy())
b = tf.constant([[1,2,3],[4,5,6]], dtype=tf.float32)
print('b:', b.numpy())
c = a+b
print('c:', c.numpy())

d = tf.range(5) # [0,1,2,3,4]
print(d)

e = tf.zeros([2,3]) # 2X3 - initialized to zero
print(e)

f = tf.ones([2,3]) # 2X3 - initialized to one
print(f)

g = tf.fill([2,3],5)
print(g)