import torch 
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from config.config import ModelConfig, TrainConfig, PathsConfig
from models_manager.models import ModelBuilder
from metrics_manager.metrics_manager import CalculateMetrics, Metrics, single_epoch_metric_dump

class Trainer:
    """
    Zarządza pełnym cyklem trenowania i walidacji modelu zgodnie z konfiguracją.
    """
    def __init__(self, model_cfg: ModelConfig, 
                 train_cfg: TrainConfig, 
                 path_cfg: PathsConfig,
                 train_loader: DataLoader, 
                 val_loader: DataLoader):
        
        """
        Buduje model/optimizer/loss, zapisuje konfigurację i dataloadery, ustawia device.
        """
        
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.paths = path_cfg

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.metrics_accumulate = Metrics()
        self.metrics_calc = CalculateMetrics()        

    
        builder = ModelBuilder(model_cfg)
        self.model, self.loss_fn, self.optimizer = builder._build()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.gpu_augment = self._build_gpu_augment()

    def _train_one_epoch(self, current_epoch: int):
        """
        Wykonuje jedną epokę treningu na `train_loader`, zwraca średni loss epoki.
        """
        BATCHES = 0
        total_loss = 0.
    
        for batch_index, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if self.gpu_augment is not None:
                inputs = self.gpu_augment(inputs)
            #call funkcji z aug
            self.optimizer.zero_grad()

            outputs = self.model(inputs) #pracujemy na logitach

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item() #kumulacja lossu z kazdego batcha 


            BATCHES += 1 
            #zwrot lossu z calej epoki 
        return total_loss / BATCHES if BATCHES else 0.0
    
    #self.epoch_number = 0

    
    def _train_loop(self):
        """
        Pętla trenowania po epokach: trening, walidacja, scheduler.step (jeśli jest), zapis najlepszego modelu.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0
        best_vloss = 1_000_000
        #if na podstawie konfigu 
        scheduler = self._build_scheduler()

        for epoch in range(self.train_cfg.epochs):
            print("Currently running Epoch no. {}:".format(epoch_number+1))

            #training
            self.model.train(True)
            avg_loss = self._train_one_epoch(current_epoch=epoch_number)

            #validation
            running_vloss = 0.0
            self.model.eval()
            self.metrics_accumulate.reset()

            with torch.no_grad():
                for i, vdata in enumerate(self.val_loader):
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)

                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss.item()
                    self.metrics_accumulate.update(voutputs, vlabels)


            avg_vloss = running_vloss / (i + 1)

            #calc and dump metrics
            v_logits, v_targets = self.metrics_accumulate.get_logits_targets()

            metrics = self.metrics_calc.get_single_label_metrics(logits=v_logits, 
                                                                 targets=v_targets)
            row = {"epoch": epoch+1, 
                   "train_loss": avg_loss, 
                   "val_loss": avg_vloss, 
                   "lr": self.optimizer.param_groups[0]["lr"], 
                   **metrics
                   }
            single_epoch_metric_dump(jsonl_path=self.paths.metrics_json_path, 
                                     row=row)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_vloss)
                else:
                    scheduler.step()

            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}.pt'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

    


    def _build_scheduler(self):
        """
        Tworzy scheduler wg `train_cfg` (cosine/step/plateau); zwraca scheduler lub None.
        """
        scheduler_name = self.train_cfg.scheduler_name.lower()

        if self.train_cfg.use_scheduler:

            if scheduler_name == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, 
                                                                 T_max=self.train_cfg.cosine_t_max)

            elif scheduler_name == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, 
                                                      step_size=self.train_cfg.step_size_lr,
                                                      gamma=self.train_cfg.gamma)

            #elif scheduler_name == "plateau":
            #    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
            #                                                     mode = self.train_cfg.mode,
            #                                                     factor = self.train_cfg.factor,
            #                                                     patience=self.train_cfg.patiance)

            else:
                raise ValueError(f"Unkown scheduler: {scheduler_name} was given!")
        else:
            return None

        return scheduler

    def _build_gpu_augment(self):
        """
        Silna augmentacja wykonywana na GPU (operuje na batchu tensorów BCHW).
        Zwróć None, jeśli chcesz ją wyłączyć.
        """
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomApply([v2.RandomRotation(degrees=25)], p=0.7),
            v2.RandomApply([v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08)], p=0.6),
            v2.RandomApply([v2.RandomPerspective(distortion_scale=0.5, p=1.0)], p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.3),
        ])
