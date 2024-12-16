import logging
from typeguard import typechecked
import datetime
import time
import tensorflow as tf

class FewShotMetric(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, model):
        """
        :param validation_sets:
        a list of 2-tuples (validation_gen, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(FewShotMetric, self).__init__()

        self.base_model = model
        self.validation_sets = validation_sets
        self.epoch = []
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_gen in self.validation_sets:
            validation_gen.support_gen.ksupport = 1
            fewshot_model = self.base_model.get_testmodel(validation_gen.nlabels,
                                                     validation_gen.support_gen.ksupport)
            results = fewshot_model.evaluate(validation_gen, verbose=0)

            for metric, result in zip(fewshot_model.metrics_names, results):
                valuename = f"few_{validation_gen.support_gen.ksupport}_shot_{metric}"

                logging.info(f"{valuename}: {result}")
                self.history.setdefault(valuename, []).append(result)

                logs[valuename] = result

            validation_gen.support_gen.ksupport = 5
            fewshot_model = self.base_model.get_testmodel(validation_gen.nlabels,
                                                     validation_gen.support_gen.ksupport)

            results = fewshot_model.evaluate(validation_gen, verbose=0)


            for metric, result in zip(fewshot_model.metrics_names, results):
                valuename = f"few_{validation_gen.support_gen.ksupport}_shot_{metric}"

                logging.info(f"{valuename}: {result}")
                self.history.setdefault(valuename, []).append(result)

                logs[valuename] = result

class AdditionalRULValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets):
        """
        :param validation_sets:
        a list of 2-tuples (validation_gen, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalRULValidationSets, self).__init__()

        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2]:
                raise ValueError()
        self.epoch = []
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_gen, name in self.validation_sets:

            results = self.model.evaluate(validation_gen,
                                          verbose=0)

            for metric, result in zip(self.model.metrics_names, results):
                valuename = name + '_' + metric

                self.history.setdefault(valuename, []).append(result)

                logs[valuename] = result



@tf.keras.utils.register_keras_serializable(package="phm_framework")
class Recall(tf.keras.metrics.Metric):
    """
    """

    def __init__(
        self, num_classes, mode='micro', name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)

        self.mode = mode
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="true_positives", shape=(self.num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="false_negatives",
            shape=(self.num_classes,),
            initializer="zeros",
        )

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false negative statistics.

        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range
            `[0, 1]`.

        Returns:
          Update op.
        """

        if y_pred.shape[1] == 1:
            y_pred = tf.reshape(tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32), shape=tf.shape(y_pred))
        else:
            y_pred = tf.argmax(y_pred, axis=1)

        y_true = tf.reshape(y_true, shape=tf.shape(y_pred))
        for i in range(self.num_classes):
            cond = tf.equal(y_true, i)

            y_k_pred = y_pred[cond] #tf.gather(y_pred, tf.where(cond))
            true_pos = tf.reduce_sum(tf.cast(tf.equal(y_k_pred, i), tf.float32))
            false_neg = tf.math.subtract(tf.cast(tf.shape(y_k_pred)[0], tf.float32), true_pos)

            self.true_positives = self.true_positives[i].assign(self.true_positives[i] + true_pos)
            self.false_negatives = self.false_negatives[i].assign(self.false_negatives[i] + false_neg)


    @tf.autograph.experimental.do_not_convert
    def result(self):
        if self.mode == 'micro':
            result = tf.math.divide_no_nan(
                tf.reduce_sum(self.true_positives),
                tf.reduce_sum(tf.math.add(self.true_positives, self.false_negatives)),
            )
        else:
            result = tf.reduce_mean(tf.math.divide_no_nan(
                self.true_positives,
                tf.math.add(self.true_positives, self.false_negatives),
            ))

        return result

    def reset_states(self):
        self.true_positives.assign(tf.zeros(shape=(self.num_classes,)))
        self.false_negatives.assign(tf.zeros(shape=(self.num_classes,)))


    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "mode": self.mode,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="phm_framework")
class NonExclusiveRecall(tf.keras.metrics.Metric):
    """
    """

    def __init__(
        self, num_classes, mode='micro', name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)

        self.mode = mode
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            "true_positives", shape=(self.num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            "false_negatives",
            shape=(self.num_classes,),
            initializer="zeros",
        )

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false negative statistics.

        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range
            `[0, 1]`.

        Returns:
          Update op.
        """

        y_true = tf.reshape(y_true, shape=tf.shape(y_pred))
        for i in range(self.num_classes):
            cond = tf.equal(y_true[:, i], 1)

            y_k_pred = y_pred[cond][:, i] #tf.gather(y_pred, tf.where(cond))
            true_pos = tf.reduce_sum(tf.cast(tf.greater_equal(y_k_pred, 0.5), tf.float32))
            false_neg = tf.math.subtract(tf.cast(tf.shape(y_k_pred)[0], tf.float32), true_pos)

            self.true_positives = self.true_positives[i].assign(self.true_positives[i] + true_pos)
            self.false_negatives = self.false_negatives[i].assign(self.false_negatives[i] + false_neg)


    @tf.autograph.experimental.do_not_convert
    def result(self):
        if self.mode == 'micro':
            result = tf.math.divide_no_nan(
                tf.reduce_sum(self.true_positives),
                tf.reduce_sum(tf.math.add(self.true_positives, self.false_negatives)),
            )
        else:
            result = tf.reduce_mean(tf.math.divide_no_nan(
                self.true_positives,
                tf.math.add(self.true_positives, self.false_negatives),
            ))

        return result

    def reset_states(self):
        self.true_positives.assign(tf.zeros(shape=(self.num_classes,)))
        self.false_negatives.assign(tf.zeros(shape=(self.num_classes,)))


    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "mode": self.mode,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="phm_framework")
class Precision(tf.keras.metrics.Metric):
    """
    """

    def __init__(
        self, num_classes, mode='micro', name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)

        self.mode = mode
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="true_positives", shape=(self.num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="false_positives",
            shape=(self.num_classes,),
            initializer="zeros",
        )

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false negative statistics.

        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range
            `[0, 1]`.

        Returns:
          Update op.
        """

        if y_pred.shape[1] == 1:
            y_pred = tf.reshape(tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32), shape=tf.shape(y_pred))
        else:
            y_pred = tf.argmax(y_pred, axis=1)

        y_true = tf.reshape(y_true, shape=tf.shape(y_pred))
        for i in range(self.num_classes):
            cond = tf.equal(y_true, i)

            y_k_pred = y_pred[cond] #tf.gather(y_pred, tf.where(cond))
            true_pos = tf.reduce_sum(tf.cast(tf.equal(y_k_pred, i), tf.float32))

            y_k_pred2 = y_pred[tf.logical_not(cond)] #tf.gather(y_pred, tf.where(tf.logical_not(cond)))
            false_pos = tf.reduce_sum(tf.cast(tf.equal(y_k_pred2, i), tf.float32))

            self.true_positives = self.true_positives[i].assign(self.true_positives[i] + true_pos)
            self.false_positives = self.false_positives[i].assign(self.false_positives[i] + false_pos)


    @tf.autograph.experimental.do_not_convert
    def result(self):
        if self.mode == 'micro':
            result = tf.math.divide_no_nan(
                tf.reduce_sum(self.true_positives),
                tf.reduce_sum(tf.math.add(self.true_positives, self.false_positives)),
            )
        else:
            result = tf.reduce_mean(tf.math.divide_no_nan(
                self.true_positives,
                tf.math.add(self.true_positives, self.false_positives),
            ))

        return result

    def reset_states(self):
        self.true_positives.assign(tf.zeros(shape=(self.num_classes,)))
        self.false_positives.assign(tf.zeros(shape=(self.num_classes,)))

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "mode": self.mode,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable(package="phm_framework")
class NonExclusivePrecision(tf.keras.metrics.Metric):
    """
    """

    def __init__(
        self, num_classes, mode='micro', name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)

        self.mode = mode
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            "true_positives", shape=(self.num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            "false_positives",
            shape=(self.num_classes,),
            initializer="zeros",
        )

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false negative statistics.

        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range
            `[0, 1]`.

        Returns:
          Update op.
        """



        y_true = tf.reshape(y_true, shape=tf.shape(y_pred))
        for i in range(self.num_classes):
            cond = tf.equal(y_true[:, i], 1)

            y_k_pred = y_pred[cond][:, i] #tf.gather(y_pred, tf.where(cond))
            true_pos = tf.reduce_sum(tf.cast(tf.greater_equal(y_k_pred, 0.5), tf.float32))

            y_k_pred2 = y_pred[tf.logical_not(cond)] #tf.gather(y_pred, tf.where(tf.logical_not(cond)))
            false_pos = tf.reduce_sum(tf.cast(tf.greater_equal(y_k_pred2, 0.5), tf.float32))

            self.true_positives = self.true_positives[i].assign(self.true_positives[i] + true_pos)
            self.false_positives = self.false_positives[i].assign(self.false_positives[i] + false_pos)


    @tf.autograph.experimental.do_not_convert
    def result(self):
        if self.mode == 'micro':
            result = tf.math.divide_no_nan(
                tf.reduce_sum(self.true_positives),
                tf.reduce_sum(tf.math.add(self.true_positives, self.false_positives)),
            )
        else:
            result = tf.reduce_mean(tf.math.divide_no_nan(
                self.true_positives,
                tf.math.add(self.true_positives, self.false_positives),
            ))

        return result

    def reset_states(self):
        self.true_positives.assign(tf.zeros(shape=(self.num_classes,)))
        self.false_positives.assign(tf.zeros(shape=(self.num_classes,)))

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "mode": self.mode,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable(package="phm_framework")
class TimeStopping(tf.keras.callbacks.Callback):
    """Stop training when a specified amount of time has passed.

    Args:
        seconds: maximum amount of time before stopping.
            Defaults to 86400 (1 day).
        verbose: verbosity mode. Defaults to 0.
    """

    @typechecked
    def __init__(self, seconds: int = 86400, verbose: int = 0):
        super().__init__()

        self.seconds = seconds
        self.verbose = verbose
        self.stopped_epoch = None

    def on_train_begin(self, logs=None):
        self.stopping_time = time.time() + self.seconds

    def on_epoch_end(self, epoch, logs={}):
        if time.time() >= self.stopping_time:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None and self.verbose > 0:
            formatted_time = datetime.timedelta(seconds=self.seconds)
            msg = "Timed stopping at epoch {} after training for {}".format(
                self.stopped_epoch + 1, formatted_time
            )
            print(msg)

    def get_config(self):
        config = {
            "seconds": self.seconds,
            "verbose": self.verbose,
        }

        base_config = super().get_config()
        return {**base_config, **config}


