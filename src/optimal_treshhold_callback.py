from tensorflow.keras.callbacks import Callback # type: ignore
import numpy as np
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, confusion_matrix

class OptimalThresholdCallback(Callback):
    def __init__(self, val_ds, name="val"):
        super().__init__()
        self.val_ds = val_ds
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred_prob = []

        for batch_imgs, batch_labels in self.val_ds:
            pred = self.model.predict(batch_imgs, verbose=0)
            if pred.shape[1] == 2:
                pred = pred[:, 1]
            y_true.append(batch_labels.numpy())
            y_pred_prob.append(pred)

        y_true = np.concatenate(y_true, axis=0)
        y_pred_prob = np.concatenate(y_pred_prob, axis=0)

        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        youden_j = tpr - fpr
        best_thresh = thresholds[np.argmax(youden_j)]

        y_pred = (y_pred_prob >= best_thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)

        logs[f"{self.name}_ot_threshold"] = best_thresh
        logs[f"{self.name}_ot_f1"] = f1
        logs[f"{self.name}_ot_precision"] = precision
        logs[f"{self.name}_ot_sensitivity"] = sensitivity
        logs[f"{self.name}_ot_specificity"] = specificity

        print(
            f"\n[{self.name}] Epoch {epoch+1}: Thr={best_thresh:.4f}, "
            f"F1={f1:.4f}, Prec={precision:.4f}, "
            f"Sens={sensitivity:.4f}, Spec={specificity:.4f}"
        )
