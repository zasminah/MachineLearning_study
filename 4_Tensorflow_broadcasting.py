import tensorflow as tf

x = tf.constant([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
y = tf.constant([[1.,1.,1.]])
print('shape:{} {}'.format(x.get_shape(), y.get_shape()))
#shape: (3,3) (1,3)

subXY = tf.subtract(x,y)
print(subXY)