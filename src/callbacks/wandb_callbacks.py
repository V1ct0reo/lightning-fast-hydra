import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all *.py files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        self.code_dir = code_dir

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")
        for path in glob.glob(os.path.join(self.code_dir, "**/*.py"), recursive=True):
            code.add_file(path)

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True):
                ckpts.add_file(path)

        experiment.log_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"].argmax(axis=1))
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds,normalize='true')

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"val/confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix.yaml/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogSequenceConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, val_window_maker):

        self.ready = True
        self.window_maker = val_window_maker
        self.seq_id_col = self.window_maker.sequenz_identifier
        self.seq_id_list = self.window_maker.seq_id_list
        # first_frame_idx = self.window_maker.window_size - 1
        # self.seq_id_preds_targtes = pd.DataFrame(
        #     columns=['seq_id', 'predicted', 'target'], index=range(first_frame_idx, self.window_maker.num_entries))

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        # clear the old seq_pred_targets dataframe...
        if self.ready:
            self.preds = []
            self.targets = []
            first_frame_idx = self.window_maker.window_size - 1
            self.seq_id_preds_targtes = pd.DataFrame(
                columns=['seq_id', 'predicted', 'target'], index=self.seq_id_list.index)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            # self.preds.append(outputs["preds"])
            # self.targets.append(outputs["targets"])

            predicted_frames_idxs = outputs['sample_idxs'][:, -1]
            predicted_frames_idxs = predicted_frames_idxs.cpu().numpy()
            predicted_labels = outputs['preds'].argmax(axis=1)
            predicted_labels = predicted_labels.cpu().numpy()
            target_batch_labels = outputs['targets'].cpu().numpy()
            self.seq_id_preds_targtes.loc[predicted_frames_idxs] = np.array([
                self.seq_id_list[predicted_frames_idxs],  # the seq_id for this window
                predicted_labels,  # the prediction for this window
                target_batch_labels  # the target for this window
            ]).T

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = []  # np.zeros((self.num_classes))
            targets = []  # np.zeros((self.num_classes))
            for seq in sorted(self.seq_id_preds_targtes.seq_id.unique()):
                if not isinstance(seq, int):
                    continue
                seq_df = self.seq_id_preds_targtes[self.seq_id_preds_targtes['seq_id'] == seq]
                majority = seq_df.mode(axis=0, dropna=True)
                t = majority['target'].values[0]
                if not isinstance(t, int):
                    t = t[0]
                p = majority['predicted'].values[0]
                if not isinstance(p, int):
                    p = p[0]
                targets.append(t)
                preds.append(p)

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"val/seq_confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix.yaml/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


    def on_test_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.window_maker = trainer.datamodule.test_dataset.window_maker
        self.seq_id_list=self.window_maker.seq_id_list
        self.preds = []
        self.targets = []
        first_frame_idx = self.window_maker.window_size - 1
        self.seq_id_preds_targtes = pd.DataFrame(
            columns=['seq_id', 'predicted', 'target'], index=self.seq_id_list.index)


    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            # self.preds.append(outputs["preds"])
            # self.targets.append(outputs["targets"])

            predicted_frames_idxs = outputs['sample_idxs'][:, -1]
            predicted_frames_idxs = predicted_frames_idxs.cpu().numpy()
            predicted_labels = outputs['preds'].argmax(axis=1)
            predicted_labels = predicted_labels.cpu().numpy()
            target_batch_labels = outputs['targets'].cpu().numpy()
            self.seq_id_preds_targtes.loc[predicted_frames_idxs] = np.array([
                self.seq_id_list[predicted_frames_idxs],  # the seq_id for this window
                predicted_labels,  # the prediction for this window
                target_batch_labels  # the target for this window
            ]).T

    def on_test_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = []  # np.zeros((self.num_classes))
            targets = []  # np.zeros((self.num_classes))
            for seq in sorted(self.seq_id_preds_targtes.seq_id.unique()):
                if not isinstance(seq, int):
                    continue
                seq_df = self.seq_id_preds_targtes[self.seq_id_preds_targtes['seq_id'] == seq]
                majority = seq_df.mode(axis=0, dropna=True)
                t = majority['target'].values[0]
                if not isinstance(t, int):
                    t = t[0]
                p = majority['predicted'].values[0]
                if not isinstance(p, int):
                    p = p[0]
                targets.append(t)
                preds.append(p)

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"test/seq_confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix.yaml/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(preds, targets, average=None)
            r = recall_score(preds, targets, average=None)
            p = precision_score(preds, targets, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits = pl_module(val_imgs)
            preds = torch.argmax(logits, axis=-1)

            # log the images as wandb Image
            experiment.log(
                {
                    f"Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )
