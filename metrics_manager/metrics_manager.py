import torch
from pathlib import Path
import json
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, accuracy_score, roc_auc_score
)


class Metrics:
    def __init__(self):
        self.logits_history = []
        self.targets_history = []

    def update(self, logits, targets, epoch):
        if logits is None or targets is None:
            raise RuntimeError("No data gathererd for epoch: {}".format(epoch+1))
        else:
            self.logits_history.append(logits.detach().cpu())
            self.targets_history.append(targets.detach().cpu())
        
    def reset(self):
        self.logits_history = []
        self.targets_history = []

    def get_logits_targets(self):
        return( 
            torch.cat(self.logits_history, dim=0),
            torch.cat(self.targets_history, dim=0)
        )
    

class CalculateMetrics:
    def __init__(self):
        self.history = {
            "f1": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
            "auc": [],
            "predictions": [],
            "targets": [],
            "confusion_matrix": []
            }

    def get_single_label_metrics(self, logits, targets):

        assert logits.ndim == 2
        assert targets.ndim == 1    

        preds = torch.argmax(logits.detach().cpu(), dim=1)

        preds = preds.numpy()
        targets = targets.detach().cpu().numpy()

        self.history["f1"].append(
            f1_score(targets, preds, average="macro", zero_division=0)
        )
        self.history["precision"].append(
            precision_score(targets, preds, average="macro", zero_division=0)
        )
        self.history["recall"].append(
            recall_score(targets, preds, average="macro", zero_division=0)
        )
        self.history["accuracy"].append(
            accuracy_score(targets, preds)
        )

        self.history["confusion_matrix"].append(
            confusion_matrix(y_true=targets, y_pred=preds).tolist()
        )

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        try:
            
            auc = roc_auc_score(
                targets,
                probs,
                multi_class="ovr",
                average="macro"
                )
            
        except ValueError:
            auc = None

        self.history["auc"].append(auc)

        return {
            "f1": self.history["f1"][-1],
            "precision": self.history["precision"][-1],
            "recall": self.history["recall"][-1],
            "accuracy": self.history["accuracy"][-1],
            "auc": self.history["auc"][-1],
            "confusion_matrix": self.history["confusion_matrix"][-1]
        }


    def get_history(self):
        return self.history
    

def single_epoch_metric_dump(jsonl_path: str, row: dict) -> None:

    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")





    