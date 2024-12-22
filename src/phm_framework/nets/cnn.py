import tensorflow as tf
import numpy as np

def create_conv_1d_network(input_shape, num_classes):
    """
    This function defines a 1D CNN architecture suitable for RUL tasks.
    The model consists of multiple convolutional layers, followed by
    max-pooling layers for feature extraction,  and dense layers for prediction.

    Parameters:
    -----------
    input_shape : tuple
        The shape of the input data (timesteps, features). This defines the dimensions
        of the input layer.

    num_classes : int
        Number of classes of the target variable

    Returns:
    --------
    tf.keras.Model
        A compiled Keras Model with the defined architecture.
    """
    # Define the input layer
    input = tf.keras.layers.Input(input_shape)

    # Initialize the input for stacking
    x = input

    # Add 4 blocks of convolutional and pooling layers
    for i in range(4):
        # Each block contains three Conv1D layers with 64 filters and a kernel size of 3
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        # Add a MaxPooling1D layer to downsample the feature maps
        x = tf.keras.layers.MaxPooling1D(2)(x)

    # Flatten the feature maps to prepare for dense layers
    x = tf.keras.layers.Flatten()(x)
    # Fully connected layers for prediction
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    # Output layer: single neuron with ReLU activation (suitable for regression tasks)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create and return the Keras Model
    return tf.keras.models.Model(inputs=input, outputs=x)


def train(train_data, num_classes, epochs, es=True, validation_data=None):
    """
    This function trains a 1D convolutional neural network using the provided training data and
    optional validation data. It employs early stopping during training and performs an initial
    sanity check to ensure that the training loss evolves correctly before proceeding with full training.

    Parameters:
    -----------
    train_data : tuple
        A tuple `(X_train, Y_train)` containing the training data:
            - `X_train` : numpy.ndarray
                Input features for training (shape: `(num_samples, timesteps, features)`).
            - `Y_train` : numpy.ndarray
                Target values for training (shape: `(num_samples,)`).

    num_classes : int
        Number of classes of the target variable

    epochs : int
        Number of epochs for the final training phase.

    es : bool, optional, default=True
        If `True`, early stopping is applied during training. Early stopping monitors the validation loss
        and stops training if the loss does not improve for 8 consecutive epochs.

    validation_data : tuple, optional, default=None
        A tuple `(X_val, Y_val)` containing validation data:
            - `X_val` : numpy.ndarray
                Input features for validation.
            - `Y_val` : numpy.ndarray
                Target values for validation.

    Returns:
    --------
    results : History
        A Keras History object containing details about the training process, including loss and metric values.

    model : tf.keras.Model
        The trained model.


    Example:
    --------
    >>> train_data = (X_train, Y_train)
    >>> validation_data = (X_val, Y_val)
    >>> results, model = train(train_data, epochs=50, validation_data=validation_data)
    """
    # Unpack training data
    X_train, Y_train = train_data

    # Configure callbacks for early stopping if enabled
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)] if es else []

    valid_train = False
    while not valid_train:
        # Create and compile the model
        model = create_conv_1d_network(X_train.shape[1:], num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'],
                      loss='sparse_categorical_crossentropy')

        # Perform an initial short training session (3 epochs) to validate training behavior
        results = model.fit(X_train, Y_train,
                            epochs=3,
                            batch_size=128,
                            verbose=1,
                            validation_data=validation_data,
                            callbacks=callbacks)

        # Check if the loss has sufficient variance to ensure proper learning
        valid_train = np.std(results.history['loss']) > 1e-6

    # Perform full training with the specified number of epochs
    results = model.fit(X_train, Y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=1,
                        validation_data=validation_data,
                        callbacks=callbacks)

    return results, model