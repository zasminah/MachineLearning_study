import tensorflow as tf

vdata1 = tf.Variable(1.0) #<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
print(vdata1)
vdata2 = tf.Variable(tf.ones((2,)))#vector, save the node address to vdata2
print(vdata2.numpy())
vdata2 = tf.Variable(tf.ones([2,1]))#matrics
print(vdata2.numpy())
vdata2.assign(tf.zeros((2,1)))# reassign vdata2 value as 'zero'
print(vdata2.numpy())
vdata2.assign_add(tf.ones((2,1))) #accumulate, original value + new value
print(vdata2.numpy())