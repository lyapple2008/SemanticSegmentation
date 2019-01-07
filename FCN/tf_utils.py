import tensorflow as tf

def conv2d_base(x, W, bias):
    weight_init = tf.constant_initializer(W, dtype=tf.float32)
    bias_init = tf.constant_initializer(bias, dtype=tf.float32)

    filters = W.shape[-1]
    kernel_size = (W.shape[1], W.shape[2])
    conv = tf.layers.conv2d(x, filters, kernel_size, )