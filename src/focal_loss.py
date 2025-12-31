import tensorflow as tf

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        if alpha is not None:
            alpha_factor = tf.reduce_sum(y_true * alpha, axis=-1)
            return tf.reduce_mean(alpha_factor * modulating_factor * ce)

        return tf.reduce_mean(modulating_factor * ce)
    return loss
