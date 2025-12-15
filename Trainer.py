import torch 
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from config.config import ModelConfig, TrainConfig
from models_manager.models import ModelBuilder
from torch.utils.tensorboard import SummaryWriter
from metrics_manager.metrics_manager import Metrics, CalculateMetrics

class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig, train_loader: DataLoader, val_loader: DataLoader):
        
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader        

        

        builder = ModelBuilder(model_cfg)
        self.model, self.loss_fn, self.optimizer = builder._build()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.metrics = Metrics()
        self.calculate_metrics = CalculateMetrics()

    def _train_one_epoch(self, current_epoch: int):
        BATCHES = 0
        total_loss = 0.
    
        for batch_index, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs) #pracujemy na logitach

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item() #kumulacja lossu z kazdego batcha 


            BATCHES += 1 
            #zwrot lossu z calej epoki 
        return total_loss / BATCHES
    
    #self.epoch_number = 0

    
    def _train_loop(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        best_vloss = 1_000_000
        scheduler = self._build_scheduler()

        for epoch in range(self.train_cfg.epochs):
            print("Currently running Epoch no. {}:".format(epoch_number+1))

            self.model.train(True)
            avg_loss = self._train_one_epoch(current_epoch=epoch_number)
            running_vloss = 0.0
            
            self.model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(self.val_loader):
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)

                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss.item()

                    self.metrics.update(logits = voutputs, targets= vlabels, epoch=epoch)
            
            avg_vloss = running_vloss / (i + 1)
            scheduler.step() #dla reduce on plateu jest inaczej daje sie avg_loss do args

            logits_epoch, targets_epoch = self.metrics.get_logits_targets()
            self.calculate_metrics.get_single_label_metrics(logits=logits_epoch, targets=targets_epoch)

            self.metrics.reset()

            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

    


    def _build_scheduler(self):

        scheduler_name = self.train_cfg.scheduler_name.lower()

        if self.train_cfg.use_scheduler:

            if scheduler_name == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, 
                                                                 T_max=self.train_cfg.cosine_t_max)

            elif scheduler_name == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, 
                                                      step_size=self.train_cfg.step_size_lr,
                                                      gamma=self.train_cfg.gamma)

            elif scheduler_name == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                 mode = self.train_cfg.mode,
                                                                 factor = self.train_cfg.factor,
                                                                 patience=self.train_cfg.patiance)

            else:
                raise ValueError(f"Unkown optimizer: {scheduler_name} was given!")
        else:
            return None

        return scheduler
