from tensorflow.keras.callbacks import Callback # type: ignore
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, accuracy_score, roc_auc_score
)

class ValidationMetricsCallback(Callback):
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
                pred_prob_class1 = pred[:, 1]
            else:
                pred_prob_class1 = pred

            y_true.append(batch_labels.numpy())
            y_pred_prob.append(pred_prob_class1)

        y_true = np.concatenate(y_true, axis=0)
        y_pred_prob = np.concatenate(y_pred_prob, axis=0)

        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true_int = np.argmax(y_true, axis=1)
        else:
            y_true_int = y_true

        y_pred = (y_pred_prob >= 0.5).astype(int)

        f1 = f1_score(y_true_int, y_pred)
        precision = precision_score(y_true_int, y_pred)
        sensitivity = recall_score(y_true_int, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true_int, y_pred).ravel()
        specificity = tn / (tn + fp)
        categorical_accuracy = accuracy_score(y_true_int, y_pred)
        try:
            auc = roc_auc_score(y_true_int, y_pred_prob)
        except ValueError:
            auc = float("nan")

        logs[f"{self.name}_f1"] = f1
        logs[f"{self.name}_precision"] = precision
        logs[f"{self.name}_sensitivity"] = sensitivity
        logs[f"{self.name}_specificity"] = specificity
        logs[f"{self.name}_categorical_accuracy"] = categorical_accuracy
        logs[f"{self.name}_auc"] = auc

        print(
            f"\n[{self.name}] Epoch {epoch+1}: "
            f"F1={f1:.4f}, Prec={precision:.4f}, "
            f"Sens={sensitivity:.4f}, Spec={specificity:.4f}, "
            f"Acc={categorical_accuracy:.4f}, AUC={auc:.4f}"
        )
