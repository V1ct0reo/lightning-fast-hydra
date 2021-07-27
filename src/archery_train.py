import json
import signal
import sys
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.callbacks.status_callback import StatusUpdateCallback
from src.status.status import TrainJobStatus
from src.utils import utils

from src.status.status import StatusEnum

log = utils.get_logger(__name__)





def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)


    # Handle status stuff....
    log.info(f"Looking for status Files")
    expected_path_for_status_file = Path(config.status_file)
    if expected_path_for_status_file.exists():
        log.info(f'Status found @ {expected_path_for_status_file}')
        try:
            status = TrainJobStatus.load_status(expected_path_for_status_file)
            log.info('Status loaded.')
        except json.decoder.JSONDecodeError as e:
            log.error('!Decoding error while trying to read status!', e)
        except Exception as e:
            log.error('!Severe error while trying to read status!', e)
            raise SystemExit()
    else:
        log.info(f'creating new Status-file @ {expected_path_for_status_file}')
        status = TrainJobStatus(
            status_path=str(expected_path_for_status_file),
            ckpt_dir=None,
            log_dir=config.log_dir,
            data_set=config.data_dir,
            hw_devices_used=[]
        )
        status.save(expected_path_for_status_file)

    if status.status == StatusEnum.FINISHED:
        log.error('this job is marked as FINISHED. Exiting now...')
        sys.exit(0)
    if not status.get_latest_ckpt_path():
        config.trainer.resume_from_checkpoint = None
        ckpt_dir = config.callbacks.model_checkpoint.dirpath
        status.set_ckpt_dir(ckpt_dir)
        status.set_latest_ckpt_path(Path(ckpt_dir).joinpath('last.ckpt'))
    else:
        config.trainer.resume_from_checkpoint = str(status.latest_ckpt_path)

    def trap_signals():
        signal.signal(signal.SIGINT, _handle_kill)
        signal.signal(signal.SIGTERM, _handle_kill)

    def _handle_kill(signal_number, _frame):
        log.warning('=GETTING KILLED=')
        try:
            status.get_killed()
        except Exception as e:
            log.exception('couldnt notify status of kill')
        signal_name = {int(s): s for s in signal.Signals}[signal_number].name
        log.warning("aborting because of %s signal" % signal_name)
        raise SystemExit("aborting because of %s signal" % signal_name)

    trap_signals()

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    # dynamically set:
    #  - n_featurs (input_dim)
    #  - seq_len
    config.model.n_features = datamodule.num_features
    config.model.seq_len = datamodule.window_size
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    callbacks.append(StatusUpdateCallback(status,config.trainer.max_epochs))
    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
        # , _convert_="partial"#TODO partial does not work. apparently a circular import
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    remaining_epochs = config.trainer.max_epochs - max(status.epochs, 0)
    if remaining_epochs > 0:
        log.info(f"Starting training for {remaining_epochs} epochs!")
        try:
            with status.training_active():  # creating a session with my status chenaniganze. Basically ensures that status gets updated
                trainer.fit(model=model, datamodule=datamodule)

        except RuntimeError as e:
            if "out of memory" in "".join(e.args):
                log.exception('--==CUDA OOM!==--', e)
                status.set_cudaoom()
            else:
                log.exception(e)

    # -----------------------------
    #  TEST
    # -----------------------------

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        # TODO check if the best model was loaded if process got killed right before testing.
        trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]