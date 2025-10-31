"""Trainer class for ECoG/EEG classification models."""

import logging
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import MetricCollection
from torch.utils.data import DataLoader

_EVALUATE_OUTPUT = List[Dict[str, float]]
log = logging.getLogger('ecog_training')


def classification_metrics(metric_list: List[str], num_classes: int) -> MetricCollection:
    """
    Create a collection of classification metrics.

    Parameters
    ----------
    metric_list : list of str
        List of metric names to include
    num_classes : int
        Number of classes

    Returns
    -------
    MetricCollection
        Collection of metrics
    """
    allowed_metrics = ['precision', 'recall', 'f1score', 'accuracy', 'matthews', 'auroc', 'kappa']
    for metric in metric_list:
        if metric not in allowed_metrics:
            raise ValueError(
                f"{metric} is not allowed. Allowed metrics: {allowed_metrics}"
            )

    metric_dict = {
        'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=1),
        'precision': torchmetrics.Precision(task='multiclass', average='macro', num_classes=num_classes),
        'recall': torchmetrics.Recall(task='multiclass', average='macro', num_classes=num_classes),
        'f1score': torchmetrics.F1Score(task='multiclass', average='macro', num_classes=num_classes),
        'matthews': torchmetrics.MatthewsCorrCoef(task='multiclass', num_classes=num_classes),
        'auroc': torchmetrics.AUROC(task='multiclass', num_classes=num_classes),
        'kappa': torchmetrics.CohenKappa(task='multiclass', num_classes=num_classes)
    }
    metrics = [metric_dict[name] for name in metric_list]
    return MetricCollection(metrics)


class ClassifierTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for EEG/ECoG classification.

    Handles training, validation, and testing with configurable metrics
    and optimization settings.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        devices: int = 1,
        accelerator: str = "cpu",
        verbose: bool = True,
        metrics: List[str] = ["accuracy"],
        optimizer: str = "adam"
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : nn.Module
            Model to train
        num_classes : int
            Number of output classes
        lr : float
            Learning rate
        weight_decay : float
            Weight decay for regularization
        devices : int
            Number of devices to use
        accelerator : str
            Accelerator type (cpu, gpu, mps, auto)
        verbose : bool
            Whether to log metrics during training
        metrics : list of str
            List of metric names to compute
        optimizer : str
            Optimizer type (adam, adamw, sgd)
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.devices = devices
        self.accelerator = accelerator
        self.metrics = metrics
        self.optimizer_name = optimizer
        self.ce_fn = nn.CrossEntropyLoss()
        self.verbose = verbose
        self.init_metrics(metrics, num_classes)

    def init_metrics(self, metrics: List[str], num_classes: int) -> None:
        """Initialize metric trackers."""
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        self.train_metrics = classification_metrics(metrics, num_classes)
        self.val_metrics = classification_metrics(metrics, num_classes)
        self.test_metrics = classification_metrics(metrics, num_classes)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 300,
        *args,
        **kwargs
    ) -> Any:
        """
        Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        max_epochs : int
            Maximum number of epochs
        *args, **kwargs
            Additional arguments for pl.Trainer

        Returns
        -------
        Any
            Training result from Trainer.fit()
        """
        trainer = pl.Trainer(
            devices=self.devices,
            accelerator=self.accelerator,
            max_epochs=max_epochs,
            *args,
            **kwargs
        )
        return trainer.fit(self, train_loader, val_loader)

    def test(
        self,
        test_loader: DataLoader,
        *args,
        **kwargs
    ) -> _EVALUATE_OUTPUT:
        """
        Test the model.

        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
        *args, **kwargs
            Additional arguments for pl.Trainer

        Returns
        -------
        list of dict
            Test results
        """
        trainer = pl.Trainer(
            devices=self.devices,
            accelerator=self.accelerator,
            *args,
            **kwargs
        )
        return trainer.test(self, test_loader)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x, *args, **kwargs)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Parameters
        ----------
        batch : tuple
            Batch of data (x, y)
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)

        if self.verbose:
            self.log("train_loss", self.train_loss(loss),
                     prog_bar=True, on_epoch=False, logger=False, on_step=True)
            for i, metric_value in enumerate(self.train_metrics.values()):
                self.log(f"train_{self.metrics[i]}", metric_value(y_hat, y),
                         prog_bar=True, on_epoch=False, logger=False, on_step=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        if self.verbose:
            self.log("train_loss", self.train_loss.compute(),
                     prog_bar=False, on_epoch=True, on_step=False, logger=True)
            for i, metric_value in enumerate(self.train_metrics.values()):
                self.log(f"train_{self.metrics[i]}", metric_value.compute(),
                         prog_bar=False, on_epoch=True, on_step=False, logger=True)
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        Parameters
        ----------
        batch : tuple
            Batch of data (x, y)
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)
        self.val_loss.update(loss)
        self.val_metrics.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if self.verbose:
            self.log("val_loss", self.val_loss.compute(),
                     prog_bar=False, on_epoch=True, on_step=False, logger=True)
            for i, metric_value in enumerate(self.val_metrics.values()):
                self.log(f"val_{self.metrics[i]}", metric_value.compute(),
                         prog_bar=False, on_epoch=True, on_step=False, logger=True)
        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step.

        Parameters
        ----------
        batch : tuple
            Batch of data (x, y)
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = self.ce_fn(y_hat, y)
        self.test_loss.update(loss)
        self.test_metrics.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        if self.verbose:
            self.log("test_loss", self.test_loss.compute(),
                     prog_bar=False, on_epoch=True, on_step=False, logger=True)
            for i, metric_value in enumerate(self.test_metrics.values()):
                self.log(f"test_{self.metrics[i]}", metric_value.compute(),
                         prog_bar=False, on_epoch=True, on_step=False, logger=True)
        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer."""
        parameters = list(self.model.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))

        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                trainable_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        return optimizer
