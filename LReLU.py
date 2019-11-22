from keras import backend as K
import tensorflow as tf
def leaky_relu(x):
    return K.tf.where(K.tf.less(x,0), 0.01*x, x)
