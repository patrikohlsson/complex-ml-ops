import tensorflow as tf

ke = tf.keras


def dense(x, dim, norm=False, actf=None, name=None, training=True):
    with tf.name_scope(name, 'dense', [x, dim, norm, actf]) as scope:
        with tf.variable_scope(scope):
            x = tf.layers.dense(x, dim, use_bias=not norm)
            if norm:
                x = tf.layers.batch_normalization(x, training=training)
            if actf != None:
                x = actf(x)
    return x


def cdense(x, dim, norm=False, actf=None, complex_activation=False, name=None, use_bias=False, training=True):
    indim = x.shape.as_list()[-1]
    with tf.name_scope(name, 'c_dense', [x, dim, norm, actf]) as scope:
        with tf.variable_scope(scope):
            w_r = tf.get_variable('kernel_r', shape=[indim, dim])
            w_i = tf.get_variable('kernel_i', shape=[indim, dim])
            w = tf.complex(w_r, w_i)
            tf.add_to_collection('kernels', w)
            x = tf.tensordot(x, w, [[-1], [0]])
            if norm:
                x = tf.complex(tf.nn.softplus(
                    tf.layers.batch_normalization(tf.abs(x), training=training)), 0.)
                x = x * tf.exp(tf.complex(0., tf.angle(x)))
            elif use_bias:
               b_r = tf.get_variable('bias_r', shape=[dim])
               b_i = tf.get_variable('bias_i', shape=[dim])
               b = tf.complex(b_r, b_i)
               x = x + b
            if actf != None:
                if complex_activation:
                    x = tf.complex(actf(tf.real(x)), actf(tf.imag(x)))
                else:
                    x = tf.complex(actf(tf.abs(x)), 0.) * \
                        tf.exp(tf.complex(0., tf.angle(x)))
    return x


class CDense(ke.layers.Layer):
    def __init__(self, output_dim, real_input=False, polar_mode=False, use_bias=True, kernel_initializer=None, **kwargs):
        self.output_dim = output_dim
        self.real_input = real_input
        self.polar_mode = polar_mode
        self.kernel_initializer = kernel_initializer
        super(CDense, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.polar_mode:
            real_initializer = ke.initializers.RandomNormal(stddev=0.001)
            imag_initializer = ke.initializers.RandomUniform(-3.14, 3.14)
        else:
            if self.kernel_initializer != None:
                real_initializer = self.kernel_initializer
                imag_initializer = self.kernel_initializer
            else:
                real_initializer = ke.initializers.VarianceScaling()
                imag_initializer = ke.initializers.VarianceScaling()
        self.real_kernel = self.add_weight(name='real_kernel',
                                           shape=(
                                               input_shape[-1].value, self.output_dim),
                                           initializer=real_initializer,
                                           dtype=tf.float32,
                                           trainable=True)
        self.imag_kernel = self.add_weight(name='imag_kernel',
                                           shape=(
                                               input_shape[-1].value, self.output_dim),
                                           initializer=imag_initializer,
                                           dtype=tf.float32,
                                           trainable=True)
        self.real_bias = self.add_weight(name='real_bias',
                                         shape=(self.output_dim,),
                                         dtype=tf.float32,
                                         initializer='zero',
                                         trainable=True)
        self.imag_bias = self.add_weight(name='imag_bias',
                                         shape=(self.output_dim,),
                                         dtype=tf.float32,
                                         initializer='zero',
                                         trainable=True)
        super(CDense, self).build(input_shape)

    def call(self, x):
        if self.real_input:
            x = tf.complex(x, 0.)
        if self.polar_mode:
            w = tf.complex(self.real_kernel, 0.) * \
                tf.exp(tf.complex(0., self.imag_kernel))
            y = tf.tensordot(x, w, [[-1], [0]])
            y = y + tf.complex(self.real_bias, 0.) * \
                tf.exp(tf.complex(0., self.imag_bias))
        else:
            w = tf.complex(self.real_kernel, self.imag_kernel)
            y = tf.tensordot(x, w, [[-1], [0]])
            y = y + tf.complex(self.real_bias, self.imag_bias)
        return y


class CActivation(ke.layers.Layer):
    def __init__(self, activation, use_magnitude=False, **kwargs):
        self.activation = activation
        self.use_magnitude = use_magnitude
        super(CActivation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CActivation, self).build(input_shape)

    def call(self, x):
        if self.use_magnitude:
            y_mag = tf.abs(x)
            y_phase = tf.angle(x)
            y_mag = self.activation(y_mag)
            y = tf.complex(y_mag, 0.) * tf.exp(tf.complex(0., y_phase))
        else:
            y = tf.complex(self.activation(tf.real(x)),
                           self.activation(tf.imag(x)))
        return y


class CBatchNormalization(ke.layers.Layer):
    def __init__(self, use_magnitude=False, **kwargs):
        self.use_magnitude = use_magnitude
        self.batch_norm_real = ke.layers.BatchNormalization()
        self.batch_norm_imag = ke.layers.BatchNormalization()
        super(CBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CBatchNormalization, self).build(input_shape)

    def call(self, x, training=None):
        if self.use_magnitude:
            y_mag = tf.abs(x)
            y_phase = tf.angle(x)
            y_mag = self.batch_norm_real(y_mag)
            y = tf.complex(y_mag, 0.) * tf.exp(tf.complex(0., y_phase))
        else:
            y = tf.complex(self.batch_norm_real(tf.real(x)),
                           self.batch_norm_imag(tf.imag(x)))
        return y
