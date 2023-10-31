import logging
from typing import Any, Dict, List, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from ..models import CentersLoss
import os
import time

_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader


def get_logger( path:str= None, name:str = None ):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(name) if name else logging.getLogger(__name__)
    if path:
        os.makedirs( path , exist_ok=True)  
    else: 
        path = "./log"
        os.makedirs( path, exist_ok=True) 
    logger.setLevel(level=logging.DEBUG)

    timeticks = time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.localtime())
    file_handler = logging.FileHandler(
        os.path.join( path , f'{timeticks}.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    file_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(file_handler)
    return logger 


def classification_metrics(metric_list: List[str], num_classes: int):
    allowed_metrics = ['precision', 'recall', 'f1score', 'accuracy']

    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Please choose 'precision', 'recall', 'f1_score', 'accuracy'"
            )
    metric_dict = {
        'accuracy':
        torchmetrics.Accuracy(task='multiclass',
                              num_classes=num_classes,
                              top_k=1),
        'precision':
        torchmetrics.Precision(task='multiclass',
                               average='macro',
                               num_classes=num_classes),
        'recall':
        torchmetrics.Recall(task='multiclass',
                            average='macro',
                            num_classes=num_classes),
        'f1score':
        torchmetrics.F1Score(task='multiclass',
                             average='macro',
                             num_classes=num_classes)
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class ClassifierTrainer(pl.LightningModule):
    r'''
        A generic trainer class for EEG classification.

        .. code-block:: python

            trainer = ClassifierTrainer(model)
            trainer.fit(train_loader, val_loader)
            trainer.test(test_loader)

           
        
        Args:
            model (nn.Module): The classification model,make sure its ouput dimension align to num_class.
            num_classes (int, optional): The number of categories in the dataset. If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. (defualt: :obj:`None`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy'. (default: :obj:`["accuracy"]`)

        .. automethod:: fit
        .. automethod:: test
    '''

    def __init__(self,
                 model: nn.Module,
                 num_classes: int = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = ["accuracy"],
                 log_path:str = None,
                 log_name:str = None
                 ):

        super().__init__()
        
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics
        self.ce_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.__try_init_num_classes()
        self.logger_ = get_logger(log_path,log_name)
    
    def __try_init_num_classes(self):
        if not self.num_classes:
            try:
                num_classes = self.model.num_classes
                self.init_num_classes(num_classes)
            except AttributeError:
                pass
        else:
            self.init_num_classes(self.num_classes)
        
    
    def init_num_classes(self,num_classes):
        self.num_classes = num_classes
        self.init_metrics(self.metrics, num_classes)


    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def get_MissingParameter(self,dataloader):
        if not self.num_classes:
            with torch.no_grad():
                x,y = next(iter(dataloader))
                n_class =  self.model(x).shape[-1]
                self.init_num_classes(n_class)

    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int = 300,
            *args,
            **kwargs) -> Any:
        r'''
        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (:obj:`torch.utils.data.dataloader.DataLoader`, :obj:`torch_geometric.loader.DataLoader`, etc).
            max_epochs (int): Maximum number of epochs to train the model. (default: :obj:`300`)
        '''
        self.get_MissingParameter(val_loader)
            
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             max_epochs=max_epochs,
                             *args,
                             **kwargs)
        return trainer.fit(self, train_loader, val_loader)

    def test(self, test_loader: DataLoader, *args,
             **kwargs) -> _EVALUATE_OUTPUT:
        r'''
        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        self.get_MissingParameter(test_loader)
        
        trainer = pl.Trainer(devices=self.devices,
                             accelerator=self.accelerator,
                             *args,
                             **kwargs)
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        # log to prog_bar
        self.log("train_loss",
                 self.train_loss(loss),
                 prog_bar=True,
                 on_epoch=False,
                 logger=False,
                 on_step=True)

        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value(y_hat, y),
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_loss",
                 self.train_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.train_metrics.values()):
            self.log(f"train_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("train_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')
        self.logger_.info(str)

        # reset the metrics
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss",
                 self.val_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.val_metrics.values()):
            self.log(f"val_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("val_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')
        self.logger_.info(str)

        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss",
                 self.test_loss.compute(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False,
                 logger=True)
        for i, metric_value in enumerate(self.test_metrics.values()):
            self.log(f"test_{self.metrics[i]}",
                     metric_value.compute(),
                     prog_bar=False,
                     on_epoch=True,
                     on_step=False,
                     logger=True)

        # print the metrics
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("test_"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')
        self.logger_.info(str)

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.Adam(trainable_parameters,
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer

    def predict_step(self,
                     batch: Tuple[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        return y_hat


class CLossClassifierTrainer(ClassifierTrainer):
    r'''
        A trainer trains classification model whose loss function contains Center loss function. PLease refer to the following infomation to comprehend how the center loss works.

        - Paper: FBMSNet: A Filter-Bank Multi-Scale Convolutional Neural Network for EEG-Based Motor Imagery Decoding
        - URL: https://ieeexplore.ieee.org/document/9837422
        - Related Project: https://github.com/Want2Vanish/FBMSNet

        
        The model structure is required to contains a decoder block which generates the deep feature code and a predictor connected to the decoder to judge which class the feature code belong to.
        Firstly, we should prepare a model which contains :obj:`decoder` , :obj:`"predict_by_feature"` attributes whose value belongs to :obj:`nn.Module` subclass.

        .. code-block:: python

            class MyModel(nn.Module):
                def __init__(self) -> None:
                    super(md,self).__init__()
                    self.decoder = nn.Linear(10,5)
                    self.predict_by_feature= nn.Linear(5,2)
    
                def forward(self,x):
                    return self.predict_by_feature(self.decoder(x))
            model = MyModel()
        
        Use :obj:`nn.ModuleDict` is a simplest way to define model.

        .. code-block:: python

            model = nn.ModuleDict( {'decoder': nn.Linear(10,5),'predict_by_feature': nn.Linear(5,2)})

                    
        And it is also feasible to define :obj:`decorder` and :obj:`predict_by_feature` method in your model.  

        .. code-block:: python

            class MyModel(nn.Module):
                def __init__(self) -> None:
                    super(md,self).__init__()
                    self.layer1 = nn.Linear(10,5)
                    self.layer2 = nn.Linear(5,2)
                
                def decoder(self,x):
                    return self.layer1(x)
                
                def predict_by_feature(feature):
                    return self.layer2(feature)
    
                def forward(self,x):
                    return self.predict_by_feature(self.decoder(x))
            model = MyModel()
        

        Then we simply pass the model to init trainer. Done!
        
        .. code-block:: python

            trainer = ClossClassifierTrainer(model)
            trainer.fit(train_loader, val_loader)
            trainer.test(test_loader)

        Args:
            model (nn.Module): model (nn.Module): The classification model, which should provide the :obj:`".decoder()"` , :obj:`".predict_by_feature()"` method or the model  should contain decoder block(nn.Module), predict block(nn.Module) attributes and ensure this two attirbutes are named "decoder","predict_by_feature".
            center_dim (int): The dimemsion of decoder output code whose mean values we can loosely regard as the "center". If you don't specify this value, trainer will automatically infer this value when we first pass dataloader into :obj:`trainer.fit()` or :obj:`trainer.test()` method. (defualt: :obj:`None`) 
            num_classes (int, optional): The number of categories in the dataset. If :obj:`None`, the number of categories will be inferred from the attribute :obj:`num_classes` of the model. (defualt: :obj:`None`)
            lammda (float): The weight of the center loss in total loss. (default: :obj:`5e-4`)
            lr (float): The learning rate. (default: :obj:`0.001`)
            weight_decay (float): The weight decay. (default: :obj:`0.0`)
            devices (int): The number of devices to use. (default: :obj:`1`)
            accelerator (str): The accelerator to use. Available options are: 'cpu', 'gpu'. (default: :obj:`"cpu"`)
            metrics (list of str): The metrics to use. Available options are: 'precision', 'recall', 'f1score', 'accuracy'. (default: :obj:`['accuracy', 'precision', 'recall', 'f1score']`)

        .. automethod:: fit
        .. automethod:: test
    '''
    def __init__(self,
                 model: nn.Module,
                 center_dim: int = None,
                 num_classes: int = None,
                 lammda: float = 0.0005,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 devices: int = 1,
                 accelerator: str = "cpu",
                 metrics: List[str] = [
                     'accuracy', 'precision', 'recall', 'f1score'],
                 log_path: str = None,
                 log_name: str = None
                 ):
        
        super(CLossClassifierTrainer,self).__init__(model, num_classes, lr, weight_decay, devices,
                             accelerator, metrics,log_path,log_name) 
        self.lammda = lammda
        self.automatic_optimization = False
        self.decoder,self.Fpredictor= (model.decoder, 
                                       model.predict_by_feature)
        self.center_dim = center_dim
        self.__try_init_center_dim()
    

        

        
        
    def __try_init_center_dim(self):
        if not self.center_dim :
            try:
                self.center_dim = self.model.center_dim
                if self.num_classes:
                    self.Closs = CentersLoss(self.num_classes, self.center_dim)
            except AttributeError:
                pass
        else:
            if self.num_classes:
                self.Closs = CentersLoss(self.num_classes, self.center_dim)

        
    
    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        # for train
        self.train_loss = torchmetrics.MeanMetric()
        self.center_loss_train = torchmetrics.MeanMetric()
        self.predict_loss_train = torchmetrics.MeanMetric()
        
        # val
        self.val_loss = torchmetrics.MeanMetric()
        self.center_loss_val = torchmetrics.MeanMetric()
        self.predict_loss_val = torchmetrics.MeanMetric()

        #test
        self.test_loss = torchmetrics.MeanMetric()
        self.center_loss_test = torchmetrics.MeanMetric()
        self.predict_loss_test= torchmetrics.MeanMetric()

        # classification metrics for train val test
        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)
    
    def reset_metric(self, state: str = "train"):
        if state == "train":
            self.train_loss.reset()
            self.center_loss_train.reset()
            self.predict_loss_train.reset()
            self.train_metrics.reset()
        elif state == "val":
            self.val_loss.reset()
            self.center_loss_val.reset()
            self.predict_loss_val.reset()
            self.val_metrics.reset()
        elif state == "test":
            self.test_loss.reset()
            self.center_loss_test.reset()
            self.predict_loss_test.reset()
            self.test_metrics.reset()
        else:
            raise ValueError("state Available options are \"train\",\"val\",\"test\"")
    
    def get_MissingParameter(self, data):
        if (not self.num_classes) or (not self.center_dim):
            with torch.no_grad():
                x,y = next(iter(data))
                f = self.decoder(x)
                self.center_dim = f.shape[-1]
                n_class =  self.Fpredictor(f).shape[-1]
                
                self.init_num_classes(n_class)
                self.Closs = CentersLoss(self.num_classes, self.center_dim)

    
    def training_step(self, batch, batch_idx) -> None:
        x, y = batch
        opt4center, opt4pred = self.optimizers(True)

        # zero_grad
        opt4center.zero_grad()
        opt4pred.zero_grad()

        # center loss
        feat = self.decoder(x)
        centerloss = self.Closs(y, feat)

        # prediction cross entropy loss
        y_hat = self.Fpredictor(feat)
        pre_loss = self.ce_fn(y_hat, y)

        # total loss
        loss = self.lammda * centerloss + pre_loss

        # backward
        self.manual_backward(loss)
  
        # step
        opt4center.step()
        opt4pred.step()
  
        # log
        log_dict = {"Train Total Loss" : self.train_loss(loss),
                    "Train CELoss" : self.predict_loss_train(pre_loss),
                    "Train CLoss": self.center_loss_train(centerloss) }

        for i, metric_value in enumerate(self.train_metrics.values()):
            log_dict[f"Train {self.metrics[i]}"] = metric_value(y_hat, y)
        self.log_dict(log_dict,
                     prog_bar=True,
                     on_epoch=False,
                     logger=False,
                     on_step=True)
        


    def on_train_epoch_end(self) -> None:

        # log loss
        log_dict = {"Train Total Loss" : self.train_loss.compute(),
                    "Train CELoss" : self.predict_loss_train.compute(),
                    "Train CLoss": self.center_loss_train.compute() }
        # log classfication metrics
        for i, metric_value in enumerate(self.train_metrics.values()):
            log_dict[f"Train {self.metrics[i]}"] = metric_value.compute()
        self.log_dict(log_dict,
                    prog_bar=True)
        
        # print the metrics
        str = "\n[Train] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("Train "):
                str += f"{key}: {value:.3f} "
        print(str + '\n')
        self.logger_.info(str)
        
        # reset the metrics
        self.reset_metric()
        
    def validation_step(self, batch: Tuple[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        x, y = batch
        # calculate feat y_pred
        feat = self.decoder(x)
        y_hat = self.Fpredictor(feat)

        # get loss (pred_loss,center loss,total_loss)
        pre_loss = self.ce_fn(y_hat, y)
        centerloss = self.Closs(y, feat)
        loss = pre_loss + self.lammda * centerloss

        # update  metrics
        self.val_loss.update(loss)
        self.center_loss_val.update(centerloss)
        self.predict_loss_val.update(pre_loss)
        self.val_metrics.update(y_hat, y)

        # log loss
        log_dict = {"Val Total Loss" : self.val_loss.compute(),
                    "Val CELoss" : self.predict_loss_val.compute(),
                    "Val CLoss": self.center_loss_val.compute() }
        # log metrics
        for i, metric_value in enumerate(self.val_metrics.values()):
            log_dict[f"Val {self.metrics[i]}"] =  metric_value.compute()
            self.log_dict(log_dict,
                          prog_bar=True,
                          on_epoch=True,
                          on_step=False,
                          logger=True)
        
        # print the metrics
        
        return loss
        

    def on_validation_epoch_end(self) -> None:
        # log loss
        log_dict = {"Val Total Loss" : self.val_loss.compute(),
                    "Val CELoss" : self.predict_loss_val.compute(),
                    "Val CLoss": self.center_loss_val.compute() }
        # log classfication metrics
        for i, metric_value in enumerate(self.val_metrics.values()):
            log_dict[f"Val {self.metrics[i]}"] = metric_value.compute()
        self.log_dict(log_dict,
                    prog_bar=True,
                    on_epoch=True,
                     on_step=False,
                     logger=True)
        # reset the metrics
        self.reset_metric("val")
        str = "\n[Val] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("Val"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')
        self.logger_.info(str)
    


    def test_step(self, batch: Tuple[torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        x, y = batch
        # centerloss
        feat = self.decoder(x) 
        centerloss = self.Closs(y,feat)

        # predict loss
        y_hat = self.Fpredictor(feat)
        pre_loss = self.ce_fn(y_hat, y)
    
        #Total loss 
        loss = pre_loss +self.lammda *centerloss

        self.test_loss.update(loss)
        self.center_loss_test.update(centerloss)
        self.predict_loss_test(pre_loss)
        self.test_metrics.update(y_hat, y)
        return loss
    
    def on_test_epoch_end(self) -> None:
        log_dict = {"Test Total Loss" : self.test_loss.compute(),
                    "Test CELoss" : self.predict_loss_test.compute(),
                    "Test CLoss": self.center_loss_test.compute() }
        # log classfication metrics
        for i, metric_value in enumerate(self.test_metrics.values()):
            log_dict[f"Test {self.metrics[i]}"] = metric_value.compute()
        self.log_dict(log_dict,
                    prog_bar=True,
                    on_epoch=True,
                     on_step=False,
                     logger=True)
        # reset the metrics
        self.reset_metric("test")
        str = "\n[Test] "
        for key, value in self.trainer.logged_metrics.items():
            if key.startswith("Test"):
                str += f"{key}: {value:.3f} "
        print(str + '\n')
        self.logger_.info(str)
        

        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters))
        optimizer4pred = torch.optim.Adam(trainable_parameters,
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        optimzer4center = torch.optim.SGD(self.Closs.parameters(), lr=0.01)
        return optimzer4center, optimizer4pred
