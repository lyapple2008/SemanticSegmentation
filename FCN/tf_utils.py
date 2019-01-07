import tensorflow as tf

def conv2d_base(x, W, bias, name):
    weight_init = tf.constant_initializer(W, dtype=tf.float32)
    bias_init = tf.constant_initializer(bias, dtype=tf.float32)

    filters = W.shape[-1]
    kernel_size = (W.shape[1], W.shape[2])
    conv = tf.layers.conv2d(x, filters, kernel_size, padding='same',
                            kernel_initializer=weight_init,
                            bias_initializer=bias_init,
                            name=name)
    return conv


def avg_pool_2x2(x):



def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))
