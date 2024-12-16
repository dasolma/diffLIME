from phm_framework.nets.ae import Autoencoder
import tensorflow as tf
from tensorflow.keras import backend as K


class VariationalAutoencoder(Autoencoder):

    def __init__(self, eta=1, alpha=0.001, loss_type='cross_entropy', **kwargs):
        super(VariationalAutoencoder, self).__init__(**kwargs)

        self.eta = eta
        self.alpha = alpha
        self.loss_type = loss_type

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def _create_latent_vector(self, encoder_input, x):

        self._mu = tf.keras.layers.Dense(self.z_dim, name='mu')(x)  # (1)
        self._log_var = tf.keras.layers.Dense(self.z_dim, name='log_var')(x)

        self._encoder_mu_log_var = tf.keras.models.Model(encoder_input, (self._mu, self._log_var))  # (2)

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        self.encoder_output = tf.keras.layers.Lambda(sampling, name='encoder_output')([self._mu, self._log_var])  # (3)

        return self.encoder_output

    def call(self, data):
        (z_mean, z_log_var), z = self._encoder_mu_log_var(data), self.encoder(data)
        reconstruction = self.decoder(z)
        data = data

        if self.loss_type == 'cross_entropy':
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
        elif self.loss_type == 'ssim':
            reconstruction_loss = 1 - tf.reduce_mean(tf.image.ssim(data, reconstruction, 2.0))
        elif self.loss_type == 'mse':
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = self.eta * reconstruction_loss + self.alpha * kl_loss

        # actualizamos las métricas
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        self.add_metric(total_loss, name='loss', aggregation='mean')
        self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
        return reconstruction

    def train_step(self, data):

        with tf.GradientTape() as tape:
            if len(data) == 2:
                data, _ = data
            (z_mean, z_log_var), z = self._encoder_mu_log_var(data), self.encoder(data)
            reconstruction = self.decoder(z)
            # data = data[0]

            if self.loss_type == 'cross_entropy':
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    )
                )
            elif self.loss_type == 'ssim':
                data = data[0]
                reconstruction_loss = 1 - tf.reduce_mean(tf.image.ssim(data, reconstruction, 2.0))
            elif self.loss_type == 'mse':
                reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = self.eta * reconstruction_loss + self.alpha * kl_loss

        # obtenemos los gradientes
        grads = tape.gradient(total_loss, self.trainable_weights)

        # actualizamos los pesos
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # actualizamos las métricas
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }