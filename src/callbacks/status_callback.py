import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback


class StatusUpdateCallback(Callback):
    def __init__(self, status, max_epochs):
        self.status = status
        from src.utils import utils
        self.logger = utils.get_logger(__name__)
        self.epoch_count = self.status.epochs
        self.max_epochs = max_epochs

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.logger.info('Fitting started!')
        try:
            # since im working on a 'no GPU system', testing f**s some stuff up
            self.status.add_hw_device(torch.cuda.get_device_name(0))
        except:
            self.logger.error('Couldnt add device to status..')
            self.logger.info('couldnt get device Name. adding CPU to status as fallback')
            self.status.add_hw_device('CPU')

    def on_train_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'LightningModule') -> None:
        self.status.epoch_finished()
        self.epoch_count += 1
        self.logger.info(
            f'Epoch {self.epoch_count}/{self.max_epochs} finished. {trainer.callback_metrics}')
        self.status.set_latest_ckpt_path(self.status.get_ckpt_dir().joinpath('last.ckpt'))

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.status.set_best_ckpt_path(trainer.checkpoint_callback.best_model_path)

    def on_fit_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.status.set_testing()
        self.logger.info('=Fitting done=')

    def on_test_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        x=5

    def on_test_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        x=6