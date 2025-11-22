import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_true_labels = tf.argmax(y_true, axis=-1)

        self.precision.update_state(y_true_labels, y_pred_labels)
        self.recall.update_state(y_true_labels, y_pred_labels)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * p * r / (p + r + 1e-7)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
