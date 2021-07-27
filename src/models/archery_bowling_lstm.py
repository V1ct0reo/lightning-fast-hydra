from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torchmetrics.classification
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F


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

    def forward(self, x) -> Any:
        x = x.view(-1, self.seq_len, self.n_features)  # (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        x = self.linear(lstm_out[:, -1])
        return x

    def configure_optimizers(self):  # TODO parameterize?
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        prediction, y_true = self._step(batch, batch_idx)
        loss = F.cross_entropy(prediction, y_true).mean()
        self.log('train_cross-ent_step:', loss)
        self.log('train_acc_step:', self.accuracy(prediction, y_true))
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        prediction, y_true = self._step(batch, batch_idx)
        loss = F.cross_entropy(prediction, y_true).mean()
        self.log('val_cross-ent_step:', loss)
        self.log('val_acc_step:', self.accuracy(prediction, y_true))

        return loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        prediction, y_true = self._step(batch, batch_idx)
        loss = F.cross_entropy(prediction, y_true).mean()
        self.log('test_cross-ent_step:', loss)
        self.log('test_acc_step:', self.accuracy(prediction, y_true))

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc_epoch', self.accuracy.compute())

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        self.log('train_acc_epoch', self.accuracy.compute())

    def on_test_epoch_end(self) -> None:
        self.log('test_acc_epoch', self.accuracy.compute())

    def _step(self, batch, batch_idx):
        x, y_true = batch
        prediction = self(x)

        return prediction, y_true
