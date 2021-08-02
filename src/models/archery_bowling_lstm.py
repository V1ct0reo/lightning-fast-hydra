import io

import hydra.utils
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Optional

import seaborn as sn
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics.classification
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F

from src.models.metrics.my_confusion_matrix_metric import MyConfusionMatrix


class SimpleLSTM(pl.LightningModule):
    def __init__(self, n_features, hidden_size=100, n_layers: int = 1, seq_len=24, dropout=0, n_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.n_features, self.hidden_size, self.n_layers, self.seq_len, self.dropout, self.n_classes = n_features, hidden_size, n_layers, seq_len, dropout, n_classes
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            dropout=dropout if dropout else 0,
            num_layers=n_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, n_classes)

        self.accuracy = torchmetrics.classification.Accuracy()
        self.confusion_matrix = MyConfusionMatrix(num_classes=n_classes, compute_on_step=True)

    def forward(self, x) -> Any:
        x = x.view(-1, self.seq_len, self.n_features)  # (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        x = self.linear(lstm_out[:, -1])
        x = x.softmax(dim=-1)
        return x

    def configure_optimizers(self):  # TODO parameterize?
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        prediction, y_true, sample_idxs = self._step(batch, batch_idx)
        loss = F.cross_entropy(prediction, y_true).mean()
        self.log('train/cross-ent/step:', loss)
        self.log('train/acc/step:', self.accuracy(prediction, y_true))
        self.prev_batch = batch
        self.prev_batch_idx = batch_idx

        return {'loss': loss, 'preds': prediction, 'targets': y_true}

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        self.log('train/acc/epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        prediction, y_true, sample_idxs = self._step(batch, batch_idx)
        loss = F.cross_entropy(prediction, y_true).mean()
        self.log('val/cross-ent/step:', loss)
        self.log('val/acc/step:', self.accuracy(prediction, y_true))

        return {'loss': loss, 'preds': prediction, 'targets': y_true, 'sample_idxs':sample_idxs}

    def on_validation_epoch_end(self) -> None:
        self.log('val/acc/epoch', self.accuracy.compute())

    def on_test_start(self) -> None:
        pass

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        prediction, y_true, sample_idxs = self._step(batch, batch_idx)
        loss = F.cross_entropy(prediction, y_true).mean()
        #self.confusion_matrix(prediction, y_true)
        # self.basic_sequence_confusion_matrix.add_batch(prediction, y_true, sample_idxs)
        # add predictions and targets to the test metrik
        self.log('test/cross-ent/step:', loss)
        self.log('test/acc/step:', self.accuracy(prediction, y_true))

        return {'loss': loss, 'preds': prediction, 'targets': y_true,'sample_idxs':sample_idxs}

    def on_test_epoch_end(self) -> None:
        # more details on tensorboard hparams tab and special metriks for hparam search:
        # https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html?highlight=hp_metric#logging-hyperparameters
        test_acc_ep = self.accuracy.compute()
        self.log('test/acc/epoch', test_acc_ep)
        self.log('hp_metric', test_acc_ep)
        conf_mat = self.confusion_matrix.compute_and_save_csv()
        # self.basic_sequence_confusion_matrix.compute_and_save_csv()
        # self.log('test_confusion-matrix_epoch',conf_mat)

        # compute the real test metrik

    def _step(self, batch, batch_idx):
        x, y_true, sample_idxs = batch
        prediction = self(x)

        return prediction, y_true, sample_idxs
