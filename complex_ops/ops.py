import tensorflow as tf
import numpy as np

ke = tf.keras
K = ke.backend


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


def dft_matrix(n_fft):
    w = tf.cast(tf.exp(-2. * np.pi * 1j / n_fft), tf.complex64)
    k = tf.range(n_fft, dtype=tf.float32)
    j = tf.reshape(k, [1, -1])
    j = j[:, :(n_fft//2+1)]
    k = tf.reshape(k, [-1, 1])
    W = w ** tf.complex(k * j, 0.) / \
        tf.complex(tf.sqrt(tf.to_float(n_fft)), 0.)
    return W


def calculate_frames(duration, window_length, hop_length):
    return int(np.ceil((duration - window_length) / hop_length))

class CFC(ke.layers.Layer):
    def __init__(self, units, **kwargs):
        self.fc = ke.layers.Dense(units * 2, **kwargs)
        super().__init__()

    def build(self, input_shape):
        self.fc.build(input_shape[:-1].as_list() + [input_shape[-1].value * 2])
        self._trainable_weights += self.fc.trainable_weights
        super().build(input_shape)

    def call(self, x):
        if x.dtype != 'complex64':
            x = tf.concat([x, tf.zeros_like(x)], axis=-1)
        else:
            x = tf.concat([tf.real(x), tf.imag(x)], axis=-1)
        y = self.fc(x)
        yr, yi = tf.split(y, 2, axis=-1)
        return tf.complex(yr, yi)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.fc.units // 2,)

class iCFC(ke.layers.Layer):
    def __init__(self, fc, **kwargs):
        self.fc = fc
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        if x.dtype != 'complex64':
            x = tf.concat([x, tf.zeros_like(x)], axis=-1)
        else:
            x = tf.concat([tf.real(x), tf.imag(x)], axis=-1)
        k = self.fc.fc.kernel
        k = tf.matmul(tf.matrix_inverse(tf.matmul(k, k, transpose_a=True)), k, transpose_b=True)
        y = tf.matmul(x, k)
        yr, yi = tf.split(y, 2, axis=-1)
        return tf.complex(yr, yi)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.fc.fc.kernel.shape[0].value,)

class CDense(ke.layers.Layer):
    def __init__(self, output_dim, polar_mode=False, use_bias=True, kernel_initializer=None, **kwargs):
        self.output_dim = output_dim
        self.polar_mode = polar_mode
        self.use_bias = use_bias
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
                scale = tf.rsqrt(float(self.output_dim))
                real_initializer = ke.initializers.RandomNormal(0., scale)
                imag_initializer = ke.initializers.RandomNormal(0., scale)
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
        if self.use_bias:
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
        if x.dtype != tf.complex64:
            x = tf.complex(x, 0.)
        if self.polar_mode:
            w = tf.complex(self.real_kernel, 0.) * \
                tf.exp(tf.complex(0., self.imag_kernel))
            y = tf.tensordot(x, w, [[-1], [0]])
            if self.use_bias:
                y = y + tf.complex(self.real_bias, 0.) * \
                    tf.exp(tf.complex(0., self.imag_bias))
        else:
            w = tf.complex(self.real_kernel, self.imag_kernel)
            y = tf.tensordot(x, w, [[-1], [0]])
            if self.use_bias:
                y = y + tf.complex(self.real_bias, self.imag_bias)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'polar_mode': self.polar_mode,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer
        }
        base_config = super(CDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class iCDense(ke.layers.Layer):
    def __init__(self, fc, **kwargs):
        self.fc = fc
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
    
    def _invert_kernel(self, k):
        return tf.matmul(tf.matrix_inverse(tf.matmul(k, k, transpose_a=True)), k, transpose_b=True)

    def call(self, x):
        kr = self._invert_kernel(self.fc.real_kernel)
        ki = self._invert_kernel(self.fc.imag_kernel)
        k = tf.complex(kr, ki)
        if self.fc.use_bias:
            x = x - tf.complex(self.fc.real_bias, self.fc.imag_bias)
        return tf.tensordot(x, k, [[-1], [0]])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.fc.real_kernel.shape[0].value,)

class CActivation(ke.layers.Layer):
    def __init__(self, activation, use_magnitude=False, **kwargs):
        self.activation = ke.activations.get(activation)
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

    def compute_output_shape(self, input_shape):
        return input_shape

class CFreqDense(ke.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        input_dim = input_shape[-1].value
        a = self.add_weight(
            'kernel',
            shape=(1, self.output_dim),
            dtype=tf.float32,
            initializer=ke.initializers.RandomNormal(0., 1.)
        )
        f = self.add_weight(
            'freqs',
            shape=(1, self.output_dim),
            dtype=tf.float32,
            initializer=ke.initializers.RandomNormal(0., 1.)
        )

        i = tf.reshape(tf.range(input_dim, dtype=tf.float32), [-1, 1])

        a = tf.nn.softmax(a, axis=-1)
        j = tf.nn.sigmoid(f)

        self.W = tf.complex(a, 0.) * tf.exp(tf.complex(0., 2 * np.pi * i * j))
        super().build(input_shape)

    def call(self, x):
        if x.dtype != 'complex64':
            x = tf.complex(x, 0.)
        y = tf.tensordot(x, self.W, [[-1], [0]])# / tf.complex(tf.sqrt(float(self.output_dim)), 0.)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

class CLayerNormalization(ke.layers.Layer):
    def __init__(self, use_magnitude=False, **kwargs):
        self.use_magnitude = use_magnitude
        self.eps = 1e-6
        super().__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1].value
        self.gamma_real = self.add_weight(
            'gamma_real',
            (dim,),
            dtype=tf.float32,
            initializer='ones'
        )
        if not self.use_magnitude:
            self.gamma_imag = self.add_weight(
                'gamma_real',
                (dim,),
                dtype=tf.float32,
                initializer='ones'
            )
            
            self.beta_real = self.add_weight(
                'beta_real',
                (dim,),
                dtype=tf.float32,
                initializer='zeros'
            )

            self.beta_imag = self.add_weight(
                'beta_imag',
                (dim,),
                dtype=tf.float32,
                initializer='zeros'
            )

        super().build(input_shape)

    def call(self, x, training=None):
        if x.dtype != 'complex64':
            x = tf.complex(x, 0.)
        if self.use_magnitude:
            y_mag = tf.abs(x)
            y_mean = K.mean(y_mag, axis=0, keepdims=True)
            y_std = K.std(y_mag, axis=0, keepdims=True)
            
            y_norm = y_mag / (y_std + self.eps)
            y = tf.complex(self.gamma_real * y_norm, 0.) * tf.exp(tf.complex(0., tf.angle(x)))
        else:
            y_mean = K.mean(x, axis=-1, keepdims=True)
            y_std = K.std(x, axis=-1, keepdims=True)
            y = tf.complex(self.gamma_real, self.gamma_imag) * (x - y_mean) / y_std + tf.complex(self.beta_real, self.beta_imag)

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


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

    def compute_output_shape(self, input_shape):
        return input_shape


class CDropout(ke.layers.Layer):
    def __init__(self, factor, use_magnitude=False, **kwargs):
        self.use_magnitude = use_magnitude
        self.dropout_real = ke.layers.Dropout(factor)
        self.dropout_imag = ke.layers.Dropout(factor)
        super(CDropout, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CDropout, self).build(input_shape)

    def call(self, x, training=None):
        if self.use_magnitude:
            y_mag = tf.abs(x)
            y_phase = tf.angle(x)
            y_mag = self.dropout_real(y_mag)
            y = tf.complex(y_mag, 0.) * tf.exp(tf.complex(0., y_phase))
        else:
            y = tf.complex(self.dropout_real(tf.real(x)),
                           self.dropout_imag(tf.imag(x)))
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

class CTrainableDFT(ke.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim

        super(CTrainableDFT, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k_increment = self.add_weight(name='k_increment',
                                           shape=(1, self.output_dim),
                                           dtype=tf.float32,
                                           trainable=True)
        self.i = tf.reshape(tf.range(input_shape[-1].value, dtype=tf.float32), [-1, 1])
        self.k = tf.cumsum(tf.exp(self.k_increment), axis=-1)

        self.w = tf.exp(tf.complex(0., 2 * np.pi * self.i * self.k))

        super(CTrainableDFT, self).build(input_shape)
    
    def call(self, x):
        if x.dtype == tf.float32:
            x = tf.complex(x, 0.)
        
        y = tf.tensordot(x, self.w, [[-1], [0]])
        return y
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)
