import torch 
import torch.optim as optim
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from torchvision.models import resnet50, mobilenet_v2, densenet121, ResNet50_Weights, MobileNet_V2_Weights, DenseNet121_Weights
from config.config import ModelConfig

class ModelBuilder:
    """
    ModelBuilder odpowiada za zbudowanie narzędzi wykorzystywanych w pętli treningowej.
    -> modelu: backbone, 
    -> funkcji straty,
    -> optymalizatora.
    Klasa przyjmuje argumenty z pliku konfiguracyjnego
    """
    def __init__(self, cfg: ModelConfig):
        """
        Parametry
        ---------
        cfg: ModelConfig
            Obiekt konfiguracji zawiera:
            - model_name: wybrana architektura (np.: "resnet50", "mobilenetv2", "densenet121")
            - num_classes: liczba klas na wyjściu
            - multi_label: czy rozwiązywany problem wymaga wielu etykiet dla jednej próbki
            - opt_name: nazwa optymalizatora (np.: "adam", "adamw", "sdg")
            - learning_rate: wybrana wartość współczynnika uczenia
            - weight_decay: wybrana wartość zaniku wag
        """
        self.cfg = cfg

    def _build_backbone(self) -> torch.nn.Module:
        """
        Buduje backbon'e modelu poprzez torchvision, dostosowuje ostatnią warstwę do liczby klas podanej w konfiguracji 
        """
        model_name = self.cfg.model_name


        if model_name == "resnet50":

            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            resnet_in = model.fc.in_features
            model.fc = nn.Linear(in_features=resnet_in, out_features=self.cfg.num_classes)

            return model

        elif model_name == "mobilenetv2":

            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            mobile_in = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features=mobile_in, out_features=self.cfg.num_classes)

            return model

        elif model_name == "densenet121":

            model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            dense_in = model.classifier.in_features
            model.classifier = nn.Linear(in_features=dense_in, out_features=self.cfg.num_classes)

            return model
        
        else:
            raise ValueError(f"Unkown model: {model_name} was given!")
        

        
    def _build_loss(self) ->torch.nn.Module:
        """
        Buduje funkcje straty w zależności od typu zadania:
        - single-label: BCEWithLogitsLoss
        - multi-label: CrossEntropyLoss
        """
        if self.cfg.multi_label:
            loss = nn.BCEWithLogitsLoss()

        elif not self.cfg.multi_label:
            loss = nn.CrossEntropyLoss()

        else:
            raise ValueError(f"Unkown loss function: {self.cfg.model_name} was given!")
        
        return loss
        
    def _build_optimizer(self, model) -> torch.optim.Optimizer:
        """
        Buduje ooptymalizator na podstawie pliku konfiguracyjnego i hiperparametrów.
        - Adam
        - AdamW
        - SDG
        """
        optimizer_name = self.cfg.opt_name.lower()

        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate)

        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=self.cfg.learning_rate)
        
        else:
            raise ValueError(f"Unkown optimizer: {optimizer_name} was given!")

        return optimizer
    
    
    def _build(self):
        """
        Zadaniem funkcji jest przekazanie parametrów konfiguracyjnych, a następnie
        zwrócenie elementów niezbędnych w pętli treningowej.
        """
        backbone = self._build_backbone()
        loss_fn = self._build_loss()
        optimizer = self._build_optimizer(model=backbone)
        
        return backbone, loss_fn, optimizer
    

    
