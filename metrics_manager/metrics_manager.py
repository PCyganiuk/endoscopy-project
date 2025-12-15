import torch 
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from config.config import ModelConfig, TrainConfig
from models_manager.models import ModelBuilder
from torch.utils.tensorboard import SummaryWriter
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
            torch.cat(self.logits_history),
            torch.cat(self.targets_history)
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
            "targets": []
            }

    def get_single_label_metrics(self, logits, targets):

        assert logits.ndim == 2
        assert targets.ndim == 1    

        preds = torch.argmax(logits.detach().cpu(), dim=1)

        preds = preds.numpy()
        targets = targets.detach().cpu().numpy()

        self.history["f1"].append(
            f1_score(targets, preds, average="macro")
        )
        self.history["precision"].append(
            precision_score(targets, preds, average="macro")
        )
        self.history["recall"].append(
            recall_score(targets, preds, average="macro")
        )
        self.history["accuracy"].append(
            accuracy_score(targets, preds)
        )

        probs = torch.softmax(logits, dim=1)
        self.history["auc"].append(
            roc_auc_score(
                targets,
                probs,
                multi_class="ovr",
                average="macro"
            )
        )

    def get_predictions_targets(self, logits, targets, epoch):
        
        targets = targets.detach().cpu().numpy()
        preds  = torch.argmax(logits.detach().cpu(), dim=1).numpy()
        self.history["targets"].append(targets.tolist())
        self.history["predictions"].append(preds.tolist())
        print("Saved targets and predictions for epoch: {}".format(epoch+1))

    def get_history(self):
        return self.history






    