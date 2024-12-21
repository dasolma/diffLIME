from collections import defaultdict
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt, patches
import math
try:
    from IPython.display import clear_output
except:
    pass

embedding_max_frequency = 100.0
embedding_dims = 8
noise_rates = list(np.linspace(1e-7, 0.1, 20, dtype=np.float32))

def add_conditional(x, features):
    """
    Add conditional information to the input tensor.

    Parameters:
    - x: Input tensor.
    - features: Conditional features to be added.

    Returns:
    - tf.Tensor: Tensor resulting from adding conditional information to the input.
    """
    c = tf.keras.layers.Flatten()(features)
    c = tf.keras.layers.Dense(x.shape[-1] * x.shape[-2])(c)
    c = tf.keras.layers.Reshape(target_shape=x.shape[1:])(c)
    # x = x + c
    a = tf.keras.layers.add([x, c])
    c = tf.keras.layers.concatenate([x, c, a])

    return c


def add_timeembedding(x, emb):
    """
    Add time-based embedding to the input tensor.

    Parameters:
    - x: Input tensor.
    - emb: Time-based embedding.

    Returns:
    - tf.Tensor: Tensor resulting from adding time-based embedding to the input.
    """

    c = tf.keras.layers.Dense(x.shape[-1] * x.shape[-2])(emb)
    c = tf.keras.layers.Reshape(target_shape=x.shape[1:])(c)
    # x = x + c
    a = tf.keras.layers.add([x, c])

    return a


def timestep_embedding(max_steps, embedding_size, embedding_factor):
    """
    Generate timestep embedding.

    Parameters:
    - max_steps: Maximum iteration.
    - embedding_size: Size of the embedding vectors.
    - embedding_factor: Factor controlling the shape of the embedding.

    Returns:
    - tf.Tensor: Embedding vectors for each timestep.
    """
    # [E // 2]
    logit = tf.linspace(0., 1., embedding_size // 2)
    exp = tf.pow(10, logit * embedding_factor)
    # [step]
    timestep = tf.range(1, max_steps + 1)
    # [step, E // 2]
    comp = exp[None] * tf.cast(timestep[:, None], tf.float32)
    # [step, E]
    return tf.concat([tf.sin(comp), tf.cos(comp)], axis=-1)

def get_network(input_size, widths, block_depth, num_features,
                batch_norm=False, cond=True, timesteps=False,
                embedding_size=32, embedding_factor=4, embedding_proj=32,
                embedding_layers=2):
    """
    Build a residual U-Net model for denoising time-series data.

    Parameters:
    - input_size: Size of the input time-series.
    - widths: List of integers representing the number of filters in each layer.
    - block_depth: Number of residual blocks in each down- and up-sampling block.
    - num_features: Number of meta-attributes.
    - batch_norm: Boolean, whether to use batch normalization.
    - cond: Boolean, whether to use conditional information (meta-attributes).
    - timesteps: Boolean, whether to use timestep information.
    - embedding_size: Size of the timestep embedding vectors.
    - embedding_factor: Factor controlling the shape of the timestep embedding.
    - embedding_proj: Dimensionality of the projection in the timestep embedding.
    - embedding_layers: Number of layers in the timestep embedding.

    Returns:
    - tf.keras.Model: Residual U-Net model for denoising time-series data.
    """

    # Define input layers
    noisy_samples = tf.keras.Input(shape=(input_size, 1), name='noisy_samples')

    if timesteps:
        timestep = tf.keras.Input(shape=(1,), name='time_step', dtype=tf.int32)
        embed = timestep_embedding(timesteps, embedding_size, embedding_factor)

        #embed = tf.gather(embed, timestep, tf.int32)
        embed = Gather()(embed, timestep)

        for _ in range(embedding_layers):
            embed = tf.keras.layers.Dense(embedding_proj)(embed)

            embed = Swish()(embed)

    if cond:
        features = tf.keras.Input(shape=(num_features, 1), name='features')

    # Initial convolutional layer
    x = tf.keras.layers.Conv1D(widths[0], kernel_size=1)(noisy_samples)

    skips = []  # List to store skip connections

    # Down-sampling blocks
    for width in widths[:-1]:
        if timesteps:
            x = add_timeembedding(x, embed)
        if cond:
            x = add_conditional(x, features)
        x = DownBlock(width, block_depth, batch_norm=batch_norm)([x, skips])

    # Residual blocks
    for _ in range(block_depth):
        if timesteps:
            x = add_timeembedding(x, embed)
        if cond:
            x = add_conditional(x, features)
        x = ResidualBlock(widths[-1], batch_norm=batch_norm)(x)

    # Up-sampling blocks
    for width in reversed(widths[:-1]):
        if timesteps:
            x = add_timeembedding(x, embed)
        if cond:
            x = add_conditional(x, features)
        x = UpBlock(width, block_depth, batch_norm=batch_norm)([x, skips])

    if timesteps:
        x = add_timeembedding(x, embed)
    if cond:
        x = add_conditional(x, features)

    # Output layer
    x = tf.keras.layers.Conv1D(1, kernel_size=1, kernel_initializer="zeros")(x)

    # Create the model
    if timesteps:
        if cond:
            return tf.keras.Model([noisy_samples, features, timestep], x, name="residual_unet")
        else:
            return tf.keras.Model([noisy_samples, timestep], x, name="residual_unet")

    else:
        if cond:
            return tf.keras.Model([noisy_samples, features], x, name="residual_unet")
        else:
            return tf.keras.Model([noisy_samples], x, name="residual_unet")



def sinusoidal_embedding(x):
    """
    Generate sinusoidal embeddings for a given input sequence.

    Parameters:
    - x (tf.Tensor): Input sequence.

    Returns:
    - tf.Tensor: Sinusoidal embeddings of the input sequence.
    """

    # Minimum frequency for sinusoidal embeddings
    embedding_min_frequency = 1.0

    # Calculate frequencies based on embedding dimensions
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )

    # Calculate angular speeds corresponding to frequencies
    angular_speeds = 2.0 * math.pi * frequencies

    # Generate sinusoidal embeddings by concatenating sine and cosine values
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=2
    )
    return embeddings


def ResidualBlock(width, batch_norm):
    """
    Create a residual block for a 1D convolutional neural network.

    Parameters:
    - width (int): Number of filters in the convolutional layers.
    - batch_norm (bool): Flag indicating whether to apply batch normalization.

    Returns:
    - function: A function that applies the residual block to a given input tensor.
    """

    def apply(x):
        # Check if the input width matches the specified width
        input_width = x.shape[2]
        if input_width == width:
            residual = x
        else:
            # If widths do not match, apply a 1x1 convolution to match the dimensions
            residual = tf.keras.layers.Conv1D(width, kernel_size=1)(x)

        # Apply batch normalization if specified
        if batch_norm:
            x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)

        # First convolutional layer with activation
        x = tf.keras.layers.Conv1D(
            width, kernel_size=3, padding="same", activation=tf.keras.activations.swish
        )(x)

        # Second convolutional layer
        x = tf.keras.layers.Conv1D(width, kernel_size=3, padding="same")(x)

        # Add the residual connection
        x = tf.keras.layers.Add()([x, residual])

        return x

    return apply


def DownBlock(width, block_depth, batch_norm):
    """
    Create a downsampling block for a U-Net architecture.

    Parameters:
    - width (int): Number of filters in the convolutional layers.
    - block_depth (int): Number of residual blocks to stack in the downsampling block.
    - batch_norm (bool): Flag indicating whether to apply batch normalization.

    Returns:
    - function: A function that applies the downsampling block to a given input tensor.
    """

    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            # Apply multiple residual blocks in the downsampling block
            x = ResidualBlock(width, batch_norm)(x)
            skips.append(x)

        # Apply average pooling for downsampling
        x = tf.keras.layers.AveragePooling1D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth, batch_norm):
    """
    Create an upsampling block for a U-Net architecture.

    Parameters:
    - width (int): Number of filters in the convolutional layers.
    - block_depth (int): Number of residual blocks to stack in the upsampling block.
    - batch_norm (bool): Flag indicating whether to apply batch normalization.

    Returns:
    - function: A function that applies the upsampling block to a given input tensor.
    """

    def apply(x):
        x, skips = x
        # Apply upsampling
        x = tf.keras.layers.UpSampling1D(size=2)(x)
        for _ in range(block_depth):
            # Concatenate skip connection and apply multiple residual blocks in the upsampling block
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width, batch_norm)(x)
        return x

    return apply

class Gather(tf.keras.layers.Layer):

    def call(self, embed, timestep):
        return tf.gather(embed, timestep, tf.int32)

class Swish(tf.keras.layers.Layer):

    def call(self, x):
        return tf.nn.swish(x)

class DiffusionModel(tf.keras.Model):
    def __init__(self, input_size, widths, block_depth, timesteps, num_features=17,
                 cond=True, feature_names=None, feature_loss_net=False, feature_loss=True):
        """
        Initialize a DiffusionModel.

        Args:
        - input_size (int): Size of the input samples.
        - widths (List[int]): List of integers specifying the widths of convolutional layers.
        - block_depth (int): Depth of residual blocks in the network.
        - timesteps (int): Number of diffusion steps in the model.
        - num_features (int): Number of features in the input samples.
        - cond (bool): Whether the model is conditional.
        - feature_names (List[str]): Names of the features.
        - feature_loss_net (bool): Whether to include a feature loss network.
        - feature_loss (bool): Whether to include feature loss in the total loss.
        """
        super().__init__()

        self.timesteps = timesteps
        self.timebars = np.linspace(0.00, 0.99, timesteps + 1, dtype=np.float32)
        self.input_size = input_size
        self.feature_names = feature_names
        self.num_features = num_features
        self.network = get_network(input_size, widths, block_depth, batch_norm=True,
                                   num_features=self.num_features, cond=cond,
                                   timesteps=timesteps)
        self.cond = cond
        self.internal_logs = defaultdict(lambda: [])
        self.__images = []
        self.feature_loss_net = feature_loss_net
        self.feature_loss = feature_loss

        self._plot_samples = None

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = tf.keras.metrics.Mean(name="noise_loss")
        self.sample_loss_tracker = tf.keras.metrics.Mean(name="signal_loss")
        if self.feature_loss_net:
            self.feature_loss_tracker = tf.keras.metrics.Mean(name="feature_loss")

    @property
    def metrics(self):
        if self.feature_loss_net:
            return [self.noise_loss_tracker, self.sample_loss_tracker,
                    self.feature_loss_tracker]
        else:
            return [self.noise_loss_tracker, self.sample_loss_tracker]

    def noise(self, samples, t=None):
        """
        Generate noisy samples using a diffusion process.

        Args:
        - samples (tf.Tensor): Input samples.
        - t (tf.Tensor or None): Time step indices for the diffusion process.

        Returns:
        - Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Tuple containing:
          - x_t (tf.Tensor): Noisy samples at time t.
          - x_t1 (tf.Tensor): Noisy samples at time t+1.
          - timesteps (tf.Tensor): Time step indices for the diffusion process.
          - noise (tf.Tensor): Normal noise applied to the samples.
          - noise_rates (tf.Tensor): Difference in noise levels between t+1 and t.
        """
        batch_size = tf.shape(samples)[0]
        if t is None:
            t = tf.random.uniform((batch_size, 1, 1), 0, self.timesteps - 1, dtype=tf.dtypes.int32)

        # noise levels/rates (nl)
        nl_t = tf.gather(self.timebars, t)
        nl_t_plus_1 = tf.gather(self.timebars, t + 1)

        # normal noise
        noise = tf.random.normal(shape=(batch_size, self.input_size, 1))

        # noise(t)  and noise(t+1)
        noise_t = noise * nl_t
        noise_t_plus_1 = noise * nl_t_plus_1

        # noisy_samples(t) and noisy_sample(t+1)
        x_t = samples * (1 - nl_t) + noise_t
        x_t1 = samples * (1 - nl_t_plus_1) + noise_t_plus_1

        return x_t, x_t1, t + 1, noise_t_plus_1 - noise_t, nl_t_plus_1 - nl_t

    def decode_samples(self, noisy_samples, pred_noises, noise_rates):
        """
        Return reconstructed samples
        """

        return (noisy_samples - pred_noises * noise_rates) / (1 - noise_rates)

    def denoise(self, noisy_samples, noise_rates, features, timesteps, training):
        """
        Denoise the input noisy samples using the trained network.

        Args:
        - noisy_samples (tf.Tensor): Noisy input samples.
        - noise_rates (tf.Tensor): Noise rates for each sample.
        - features (tf.Tensor or None): Additional features.
        - timesteps (tf.Tensor or None): Time step indices.
        - training (bool): Flag indicating whether the model is in training mode.

        Returns:
        - Tuple[tf.Tensor, tf.Tensor]: Tuple containing:
          - pred_noises (tf.Tensor): Predicted noise for the input samples.
          - pred_samples (tf.Tensor): Denoised samples.
        """
        network = self.network

        _input = self.prepare_inputs(noisy_samples, features, timesteps)
        pred_noises = network(_input, training=training)

        # pred_samples = self.decode_samples(noisy_samples, pred_noises, noise_rates)
        pred_samples = noisy_samples - pred_noises

        return pred_noises, pred_samples

    def prepare_inputs(self, noisy_samples, features=None, timesteps=None):
        """
        Prepare input data for the denoising network.

        Args:
        - noisy_samples (tf.Tensor): Noisy input samples.
        - features (tf.Tensor or None): Additional features.
        - timesteps (tf.Tensor or None): Time step indices.

        Returns:
        - List[tf.Tensor]: List containing the prepared input tensors for the denoising network.
        """
        _inputs = [noisy_samples]

        if self.cond:
            _inputs.append(features)

        _inputs.append(timesteps)

        return _inputs

    def neighbourhood(self, noisy_samples, noise_rates, signal_rates,
                      features=None, timesteps=None):
        """
        Perform denoising for a given number of diffusion steps.

        Args:
        - noisy_samples (tf.Tensor): Noisy input samples.
        - noise_rates (tf.Tensor): Noise rates for each sample.
        - signal_rates (tf.Tensor): Signal rates for each sample.
        - features (tf.Tensor or None): Additional features.
        - timesteps (tf.Tensor or None): Time step indices.

        Returns:
        - Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Tuple containing:
          - pred_samples (tf.Tensor): Denoised samples.
          - pred_noises (tf.Tensor): Predicted noise for the input samples.
          - features (tf.Tensor): Predicted features for the denoised samples.
        """
        _inputs = self.prepare_inputs(noisy_samples, features, timesteps)
        pred_noises = self.network(_inputs)

        pred_samples = self.decode_samples(noisy_samples, pred_noises, noise_rates)

        # features = self.feature_loss_net(pred_samples)

        return pred_samples, pred_noises

    def train_step(self, samples):
        """
        Perform a training step for the diffusion model.

        Args:
        - samples (tf.Tensor or tuple): Input samples and features if conditioning is used.

        Returns:
        - Dict[str, tf.Tensor]: Dictionary of training metrics.
        """

        if self.cond:
            samples, features = samples[0]
        else:
            if len(samples[0]) == 2:
                samples, features = samples[0]
            else:
                samples, features = samples[0], None

        batch_size = samples.shape[0]

        x_t, x_t1, timesteps, noises, noise_rates = self.noise(samples)

        with tf.GradientTape() as tape:
            # Train the network to predict the normal noise introduced
            pred_noises, pred_xt = self.denoise(
                x_t1, noise_rates, features, timesteps, training=True
            )

            noise_loss = self.loss(noises, pred_noises) / noise_rates  # used for training
            sample_loss = self.loss(x_t, pred_xt) / (1 - noise_rates)  # only used as metric

            total_loss = tf.reduce_mean(noise_loss, axis=-1)

        gradients = tape.gradient(total_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.sample_loss_tracker.update_state(sample_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, samples):
        if self.cond:
            samples, features = samples
        else:
            if len(samples[0]) == 2:
                samples, features = samples[0]
            else:
                samples, features = samples[0], None

        base_samples, noisy_samples, timesteps, noises, noise_rates = self.noise(samples)

        # use the network to separate noisy samples to their components
        pred_noises, pred_samples = self.denoise(
            noisy_samples, noise_rates, features, timesteps, training=False
        )

        noise_loss = self.loss(noises, pred_noises) / noise_rates
        sample_loss = self.loss(base_samples, pred_samples) / (1 - noise_rates)

        self.data = (base_samples, noisy_samples, timesteps, noises, noise_rates, pred_noises, pred_samples)

        self.sample_loss_tracker.update_state(sample_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        if self.feature_loss_net:
            features_pred = tf.expand_dims(self.feature_loss_net(pred_samples), axis=-1)

            if features is not None:
                feature_loss = self.loss(features, features_pred)

                self.feature_loss_tracker.update_state(feature_loss)

        return {m.name: m.result() for m in self.metrics}

    def plot_signal(self, signals, features, feature_names, ax, show_feature_names=False, legend=None):
        xlim = self.input_size

        lines = []
        for s in signals:
            l = plt.plot(s, alpha=0.7)
            lines.append(l)
        plt.axis("off")

        if legend is not None:
            ax.legend(legend)

        if features is not None:
            rect_size = xlim // len(feature_names)
            for i, (fv, a) in enumerate(zip(features, feature_names)):
                ax.add_patch(
                    patches.Rectangle(
                        xy=(rect_size * (i + 1) + 0.2, -0.1),  # point of origin.
                        width=rect_size, height=0.1, linewidth=1,
                        color='orange', fill=True, alpha=fv.numpy()))
                if show_feature_names:
                    ax.annotate(a, xy=(rect_size * i, -0.6),
                                xytext=(rect_size * i, -0.6),
                                fontsize=12,
                                rotation=60)

    def plot_loss(self, ax, values, labels, ticks, title):
        for label, values in zip(labels, values):
            label = 'val' if 'val' in label else 'train'
            label = f"{label} ({values[-1]:.3f})"
            plt.plot(range(1, len(values) + 1), values, label=label)

        plt.xticks(ticks)
        ax.title.set_text(title)
        ax.legend(loc='upper right')

    def plot_images(self, signals, features, epoch=None, logs=None, num_rows=2, num_cols=6):

        att = ['st', 'pr', 'pe', 'os', 'co', 'si', 'sl', 'in', 'pe', 'no',
               'dy', 'mn', 'mx', 'es', 'va']

        att = ['stability', 'periodicity', 'peculiarity', 'oscilatlion', 'complex',
               'simetry', 'slope', 'inform', 'peaks', 'noise', 'dyn range',
               'min val', 'max val', 'estability', 'variability', 'min_value', 'max_value']

        att = self.feature_names

        max_l = max([len(s) for s in att])
        att = [(' ' * (max_l - len(s))) + s for s in att]

        # plot random generated samples for visual evaluation of generation quality
        nsamples = num_rows * num_cols + 2;

        denoising_steps = self.timesteps
        t = np.array([denoising_steps] * signals.shape[0])
        signal_rates, noise_rates = (1 - self.timebars[t]), self.timebars[t]
        noise_rates = tf.reshape(noise_rates, (nsamples, 1, 1))
        signal_rates = tf.reshape(signal_rates, (nsamples, 1, 1))
        noise = tf.random.normal(shape=signals.shape)
        noise = noise * noise_rates

        base_signal = signals * signal_rates
        noisy_samples = (base_signal + noise).numpy()

        network = self.network

        for i in range(denoising_steps-1, 0, -1):
            t = np.array([i] * signals.shape[0])

            _input = self.prepare_inputs(noisy_samples, features, t)
            pred_noises = self.network(_input, training=False)

            noisy_samples = (noisy_samples - pred_noises).numpy()

        """
        pred_noise = None
        for i in range(denoising_steps):
            noisy_samples, pred_noise_t = self.neighbourhood(
                noisy_samples, noise_rates, signal_rates, features=features,
                timesteps=t,
            )

            if pred_noise is None:
                pred_noise = pred_noise_t * noise_rates
            else:
                pred_noise = pred_noise + pred_noise_t * noise_rates
        """

        generated_signals = noisy_samples
        show = False
        try:
            clear_output(wait=True)
            show = True
        except:
            pass

        plt.figure(figsize=(num_cols * 2.0, num_rows * 7.0))
        plt.suptitle(f'Epoch {epoch + 1}')

        ax = plt.subplot(4, 2, 1)
        self.plot_signal((noisy_samples[0],
                          signals[0],
                          generated_signals[0],
                          # noise[0],
                          # pred_noise[0],
                          ),
                         None,
                         None, ax, False,
                         legend=['noisy', 'original', 'generated', 'noise', 'pred noise'])
        ax = plt.subplot(4, 2, 2)
        self.plot_signal((noisy_samples[1],
                          signals[1],
                          generated_signals[1],
                          # noise[1],
                          # pred_noise[1],

                          ),
                         None, None, ax, False)

        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col + 12
                ax = plt.subplot(num_rows * 4, num_cols, index + 1)
                self.plot_signal((
                    noisy_samples[index - 10],
                    signals[index - 10],
                    generated_signals[index - 10],
                    # noise[index-10],
                    # pred_noise[index-10],

                ),
                    None, None, ax,
                    False)

        # show training losses
        internal_logs = defaultdict(lambda: [])
        for k, v in self.internal_logs.items():
            internal_logs[k] = v

        for k, v in logs.items():
            internal_logs[k].append(v)

        self.internal_logs = internal_logs

        ticks = range(1, epoch + 1)
        if len(ticks) > 5:
            step = len(ticks) // 5
            ticks = [t for t in ticks if t % step == 0]

        labels, values = zip(*[(k, v) for k, v in self.internal_logs.items() if 'noise' in k])
        ax = plt.subplot(4, 2, 5)
        self.plot_loss(ax, values, labels, ticks, 'Noise loss')

        labels, values = zip(*[(k, v) for k, v in self.internal_logs.items() if 'signal' in k])
        ax = plt.subplot(4, 2, 6)
        self.plot_loss(ax, values, labels, ticks, 'Signal loss')

        plt.tight_layout()

        if show:
            plt.show()

        plt.close()
