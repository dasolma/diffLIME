import tensorflow as tf

class DistanceLayer(tf.keras.layers.Layer):
    """
    """

    def __init__(self, distance_type='euclidean', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.distance_type = distance_type

    def call(self, q, s):

        if self.distance_type == 'l1':
            distance = tf.math.abs(q - s)
            return distance
        elif self.distance_type == 'l2':
            distance = tf.square(q - s)
            return distance
        elif self.distance_type == 'cosine':
            normalize_a = tf.math.l2_normalize(q, -1)
            normalize_b = tf.math.l2_normalize(s, -1)
            cos_similarity = (1 + tf.multiply(normalize_a, normalize_b)) / 2

            return cos_similarity
        elif self.distance_type == 'product':
            return tf.multiply(q, s)

    def get_config(self):
        config = {
            "distance_type": self.distance_type,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))