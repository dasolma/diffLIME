import tensorflow as tf
import inspect
import numpy as np



class Autoencoder(tf.keras.Model):

    def __init__(self, input_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
                 decoder_conv_filters, decoder_conv_kernel_size, decoder_conv_strides, activation, z_dim,
                 use_batch_normalization=False, dropout=None):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_filters = decoder_conv_filters
        self.decoder_conv_kernel_size = decoder_conv_kernel_size
        self.decoder_conv_strides = decoder_conv_strides
        self.z_dim = z_dim
        self.activation = activation
        self.use_batch_normalization = use_batch_normalization
        self.dropout = dropout

        self.encoder, encoder_input, encoder_output = self.__create_encoder()
        self.decoder = self.__create_decoder()
        self.model = tf.keras.models.Model(encoder_input, self.decoder(encoder_output))

    def __create_encoder(self):

        encoder_input = tf.keras.layers.Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input
        for i in range(len(self.encoder_conv_filters)):
            conv_layer = tf.keras.layers.Conv1D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding='same',
                activation=self.activation if isinstance(self.activation, str) else None,
                name='encoder_conv_' + str(i)
            )

            x = conv_layer(x)  # (2)

            if inspect.isclass(self.activation) and tf.keras.layers.Layer in self.activation.__bases__:
                x = self.activation()(x)

            if self.use_batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)

            if self.dropout:
                x = tf.keras.layers.Dropout(self.dropout)(x)

        self.__shape_before_flattening = tf.shape(x)._inferred_value[1:]
        x = tf.keras.layers.Flatten()(x)  # (3)

        encoder_output = self._create_latent_vector(encoder_input, x)

        return tf.keras.models.Model(encoder_input, encoder_output), encoder_input, encoder_output  # (5)

    def _create_latent_vector(self, encoder_input, x):
        encoder_output = tf.keras.layers.Dense(self.z_dim, name='encoder_output')(x)
        return encoder_output

    def __create_decoder(self):

        decoder_input = tf.keras.layers.Input(shape=(self.z_dim,), name='decoder_input')  # (1)

        x = tf.keras.layers.Dense(np.prod(self.__shape_before_flattening))(decoder_input)  # (2)

        x = tf.keras.layers.Reshape(self.__shape_before_flattening)(x)  # (3)

        for i in range(len(self.decoder_conv_filters)):
            activation = self.activation if isinstance(self.activation, str) else None
            conv_layer = tf.keras.layers.Conv1DTranspose(
                filters=self.decoder_conv_filters[i],
                kernel_size=self.decoder_conv_kernel_size[i],
                strides=self.decoder_conv_strides[i],
                padding='same',
                activation=activation if i < len(self.decoder_conv_filters) - 1 else 'linear',
                name='decoder_conv_' + str(i)
            )

            x = conv_layer(x)  # (4)

            if i < len(self.decoder_conv_filters) - 1:
                if inspect.isclass(self.activation) and tf.keras.layers.Layer in self.activation.__bases__:
                    x = self.activation()(x)

                if self.use_batch_normalization:
                    x = tf.keras.layers.BatchNormalization()(x)

                if self.dropout:
                    x = tf.keras.layers.Dropout(self.dropout)(x)

        decoder_output = x

        return tf.keras.models.Model(decoder_input, decoder_output)  # (6)

    def call(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        return self.model.compile(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.model.train_step(*args, **kwargs)


class SkipConnAutoencoder(Autoencoder):

    def __init__(self, *args, **kwards):
        super(SkipConnAutoencoder, self).__init__(*args, **kwards)

    def __create_encoder(self):

        encoder_input = tf.keras.layers.Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input
        conv_outs = [x]
        for i in range(len(self.encoder_conv_filters)):
            if len(conv_outs) > 0:
                xs = []
                for xant in conv_outs:
                    xant2 = tf.keras.layers.Conv1D(filters=x.shape[-1], kernel_size=1,
                                                   strides=int(xant.shape[-2] / x.shape[-2]),
                                                   padding='same',
                                                   activation=self.activation if isinstance(self.activation,
                                                                                            str) else None)(xant)
                    xs.append(xant2)

                x = tf.keras.layers.Concatenate()([x] + xs)

            for k in range(3):
                conv_layer = tf.keras.layers.Conv1D(
                    filters=self.encoder_conv_filters[i],
                    kernel_size=self.encoder_conv_kernel_size[i],
                    strides=1 if k < 2 else self.encoder_conv_strides[i],
                    padding='same',
                    activation=self.activation if isinstance(self.activation, str) else None,
                    name=f'encoder_conv_{i}.{k}'
                )

                x = conv_layer(x)  # (2)

            conv_outs.append(x)

            if inspect.isclass(self.activation) and tf.keras.layers.Layer in self.activation.__bases__:
                x = self.activation()(x)

            if self.use_batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)

            if self.dropout:
                x = tf.keras.layers.Dropout(self.dropout)(x)

        self.__shape_before_flattening = tf.shape(x)._inferred_value[1:]
        x = tf.keras.layers.Flatten()(x)  # (3)

        encoder_output = self._create_latent_vector(encoder_input, x)

        return tf.keras.models.Model(encoder_input, encoder_output), encoder_input, encoder_output  # (5)


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides, activation,
                 use_batch_normalization,
                 dropout, z_dim):
        super().__init__()
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.activation = activation
        self.use_batch_normalization = use_batch_normalization
        self.dropout = dropout
        self.z_dim = z_dim

        self.convolutions = {}
        self.norms = {}
        self.dense = tf.keras.layers.Dense(self.z_dim, name='encoder_output')

        for i in range(len(self.encoder_conv_filters)):
            self.convolutions[i] = []

            for k in range(3):
                conv_layer = tf.keras.layers.Conv1D(
                    filters=self.encoder_conv_filters[i],
                    kernel_size=self.encoder_conv_kernel_size[i],
                    strides=1 if k < 2 else self.encoder_conv_strides[i],
                    padding='same',
                    activation=self.activation if isinstance(self.activation, str) else None,
                )

                self.convolutions[i].append(conv_layer)

            if self.use_batch_normalization:
                self.norms[i] = tf.keras.layers.BatchNormalization()

    def call(self, inputs):

        encoder_input = inputs

        x = encoder_input

        conv_outs = [x]
        for i in range(len(self.encoder_conv_filters)):
            '''
            if len(conv_outs) > 0:
                xs = []
                factors = np.cumprod(self.encoder_conv_strides[:len(conv_outs)])[::-1]
                print(factors)
                for k, xant in enumerate(conv_outs):

                    print(xant)
                    xant2 = tf.keras.layers.MaxPooling1D((factors[k],))(xant)
                    print(xant2)

                    xant2 = tf.keras.layers.Conv1D(
                                   filters= 32,
                                   kernel_size = (1,),
                                   strides =1,
                                   padding='same', 
                                   activation=self.activation if isinstance(self.activation, str) else None) (xant2)

                    xs.append(xant2)

                x = tf.keras.layers.Concatenate()([x] + xs)
                '''
            for k in range(3):
                conv_layer = self.convolutions[i][k]

                x = conv_layer(x)  # (2)

            conv_outs.append(x)

            if inspect.isclass(self.activation) and tf.keras.layers.Layer in self.activation.__bases__:
                x = self.activation()(x)

            if self.use_batch_normalization:
                x = self.norms[i](x)

            if self.dropout:
                x = tf.keras.layers.Dropout(self.dropout)(x)

        x = tf.keras.layers.Flatten()(x)  # (3)
        encoder_output = self.dense(x)

        # print("Encoder output", encoder_output.shape)

        return encoder_output

    def compute_output_shape_before_flatten(self, input_shape):
        dim = input_shape[1]
        red = np.prod(self.encoder_conv_strides)
        return (dim // red, self.encoder_conv_filters[-1])

    def compute_output_shape(self, input_shape):
        return (None, self.z_dim)


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, decoder_conv_filters, decoder_conv_kernel_size, decoder_conv_strides, activation,
                 use_batch_normalization,
                 dropout, z_dim, shape_before_flattening):
        super().__init__()
        self.decoder_conv_filters = decoder_conv_filters
        self.decoder_conv_kernel_size = decoder_conv_kernel_size
        self.decoder_conv_strides = decoder_conv_strides
        self.activation = activation
        self.use_batch_normalization = use_batch_normalization
        self.dropout = dropout
        self.z_dim = z_dim
        self.shape_before_flattening = shape_before_flattening

        self.convolutions = {}
        self.norms = {}
        self.dense = tf.keras.layers.Dense(np.prod(self.shape_before_flattening))

        for i in range(len(self.decoder_conv_filters)):
            self.convolutions[i] = []

            activation = self.activation if isinstance(self.activation, str) else None
            conv_layer = tf.keras.layers.Conv1DTranspose(
                filters=self.decoder_conv_filters[i],
                kernel_size=self.decoder_conv_kernel_size[i],
                strides=self.decoder_conv_strides[i],
                padding='same',
                activation=activation if i < len(self.decoder_conv_filters) - 1 else 'linear',
                name='decoder_conv_' + str(i)
            )

            self.convolutions[i] = conv_layer

            if self.use_batch_normalization:
                self.norms[i] = tf.keras.layers.BatchNormalization()

    def call(self, inputs):

        x = self.dense(inputs)  # (2)

        x = tf.keras.layers.Reshape(self.shape_before_flattening)(x)  # (3)

        for i in range(len(self.decoder_conv_filters)):
            activation = self.activation if isinstance(self.activation, str) else None
            conv_layer = self.convolutions[i]
            x = conv_layer(x)  # (4)

            if i < len(self.decoder_conv_filters) - 1:
                if inspect.isclass(self.activation) and tf.keras.layers.Layer in self.activation.__bases__:
                    x = self.activation()(x)

                if self.use_batch_normalization:
                    x = self.norms[i](x)

                if self.dropout:
                    x = tf.keras.layers.Dropout(self.dropout)(x)

        decoder_output = x

        # print("Decoder output", x.shape, self.z_dim)

        return decoder_output

    def compute_output_shape(self, input_shape):
        dim = input_shape[1]
        red = np.prod(self.decoder_conv_strides)
        return (None, dim * red, self.decoder_conv_filters[-1])

class TimeDistributedAutoencoder(tf.keras.Model):

    def __init__(self, input_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
                 decoder_conv_filters, decoder_conv_kernel_size, decoder_conv_strides, activation, z_dim,
                 use_batch_normalization=False, dropout=None, slices=10):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_filters = decoder_conv_filters
        self.decoder_conv_kernel_size = decoder_conv_kernel_size
        self.decoder_conv_strides = decoder_conv_strides
        self.slices = slices
        self.z_dim = z_dim
        self.activation = activation
        self.use_batch_normalization = use_batch_normalization
        self.dropout = dropout

        self.encoder, encoder_input, encoder_output = self.__create_encoder()
        self.decoder = self.__create_decoder()
        self.model = tf.keras.models.Model(encoder_input, self.decoder(encoder_output))

    def __create_encoder(self):
        encoder_input = (self.input_dim[0] // self.slices, 1)

        model_input = tf.keras.layers.Input(shape=self.input_dim, name='model_input')

        x = tf.keras.layers.Reshape((self.slices, self.input_dim[0] // self.slices, 1))(model_input)
        # x = tf.reshape(model_input, shape=(tf.shape(model_input)[0], self.slices, self.input_dim[0] // self.slices, 1))

        encoder_layer = EncoderLayer(
            encoder_conv_filters=[128] * depth,
            encoder_conv_kernel_size=[3] * (depth),
            encoder_conv_strides=[1] + [2] * (depth - 2) + [1],
            z_dim=self.z_dim,
            activation='relu',
            use_batch_normalization=False,
            dropout=False
        )

        x = tf.keras.layers.TimeDistributed(encoder_layer)(x)

        self.__shape_before_flattening = encoder_layer.compute_output_shape_before_flatten((None,) + encoder_input)

        encoder_output = tf.keras.layers.Flatten()(x)  # (3)

        return tf.keras.models.Model(model_input, encoder_output), model_input, encoder_output  # (5)

    def __create_decoder(self):
        decoder_input = tf.keras.layers.Input(shape=(self.z_dim * self.slices,), name='decoder_input')  # (1)

        x = tf.keras.layers.Reshape((self.slices, self.z_dim))(decoder_input)

        decoder_output = tf.keras.layers.TimeDistributed(DecoderLayer(
            self.decoder_conv_filters,
            self.decoder_conv_kernel_size,
            self.decoder_conv_strides,
            self.activation,
            self.use_batch_normalization,
            self.dropout,
            self.z_dim,
            self.__shape_before_flattening,
        ))(x)

        decoder_output = tf.keras.layers.Flatten()(decoder_output)
        decoder_output = tf.expand_dims(decoder_output, axis=-1)

        return tf.keras.models.Model(decoder_input, decoder_output)  # (6)

    def call(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        tf.print("Output", output.shape)

        return output

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        return self.model.compile(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.model.train_step(*args, **kwargs)

class Fourier(tf.keras.layers.Layer):
    def __init__(self, signal_size):
        super().__init__()
        self.signal_size = signal_size

    def call(self, inputs):
        import math as m
        deg, amp, offt, offy = inputs
        offt = offt * self.signal_size
        t = tf.range(0, self.signal_size, dtype=tf.float32)
        t = tf.repeat([t], tf.shape(deg)[0], axis=0)
        t = (t + offt)

        t = tf.expand_dims(t, axis=-1)
        z = tf.expand_dims(deg * m.pi, axis=1)

        s = tf.sin(z * t)
        a = tf.expand_dims(amp, axis=1)
        s = a * s

        s = tf.reduce_sum(s, axis=-1)
        s = tf.expand_dims(offy + s, axis=-1)

        return s

class FourierAutoencoder(tf.keras.Model):

    def __init__(self, input_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
                 decoder_conv_filters, decoder_conv_kernel_size, decoder_conv_strides, activation, z_dim,
                 use_batch_normalization=False, dropout=None):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_filters = decoder_conv_filters
        self.decoder_conv_kernel_size = decoder_conv_kernel_size
        self.decoder_conv_strides = decoder_conv_strides
        self.z_dim = z_dim
        self.activation = activation
        self.use_batch_normalization = use_batch_normalization
        self.dropout = dropout

        self.encoder, encoder_input, encoder_output = self.__create_encoder()
        self.decoder = self.__create_decoder()
        self.model = tf.keras.models.Model(encoder_input, self.decoder(encoder_output))

    def __create_encoder(self):

        encoder_input = tf.keras.layers.Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input
        for i in range(len(self.encoder_conv_filters)):
            conv_layer = tf.keras.layers.Conv1D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding='same',
                activation=self.activation if isinstance(self.activation, str) else None,
                name='encoder_conv_' + str(i)
            )

            x = conv_layer(x)  # (2)

            if inspect.isclass(self.activation) and tf.keras.layers.Layer in self.activation.__bases__:
                x = self.activation()(x)

            if self.use_batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)

            if self.dropout:
                x = tf.keras.layers.Dropout(self.dropout)(x)

        self.__shape_before_flattening = tf.shape(x)._inferred_value[1:]
        x = tf.keras.layers.Flatten()(x)  # (3)

        # x = tf.keras.layers.Dense(256, name='z_output')(x)
        # x = tf.keras.layers.LeakyReLU()(x)

        degrees = tf.keras.layers.Dense(self.z_dim, name='degrees_output')(x)
        degress = tf.keras.layers.LeakyReLU()(degrees)
        amp = tf.keras.layers.Dense(self.z_dim, name='amplitud_output')(x)
        amp = tf.keras.layers.LeakyReLU()(amp)
        offt = tf.keras.layers.Dense(1, name='offt_output')(x)
        offt = tf.keras.layers.LeakyReLU()(offt)
        offy = tf.keras.layers.Dense(1, name='offy_output')(x)
        offy = tf.keras.layers.LeakyReLU()(offy)

        return tf.keras.models.Model(encoder_input, [degrees, amp, offt, offy]), encoder_input, [degrees, amp, offt,
                                                                                                 offy]  # (5)

    def __create_decoder(self):

        deg = tf.keras.layers.Input(shape=(self.z_dim,), name='deg_input')  # (1)
        amp = tf.keras.layers.Input(shape=(self.z_dim,), name='amp_input')  # (1)
        offt = tf.keras.layers.Input(shape=(1,), name='offt_input')  # (1)
        offy = tf.keras.layers.Input(shape=(1,), name='offy_input')  # (1)

        x = Fourier(self.input_dim[0])([deg, amp, offt, offy])

        decoder_output = x

        return tf.keras.models.Model([deg, amp, offt, offy], decoder_output)  # (6)

    def call(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        return self.model.compile(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.model.train_step(*args, **kwargs)

