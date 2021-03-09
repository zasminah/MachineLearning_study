import tensorflow as tf

def t_func1(x) :
    tv = tf.Variable([[4,5],[9,10]])
    return tv*x
print(t_func1(10))

@tf.function #improve compile speed
def t_func2(a,b):
    return tf.matmul(a,b)
x=[[4,5,6],[7,8,9]]
w = tf.Variable([[2,5],[6,5],[17,10]])
print(t_func2(x,w))