import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import dacite

from src.utils import utils

log = utils.get_logger(__name__)


def json_dumper(obj):
    # TODO may catch logger objects?
    if isinstance(obj, logging.Logger):
        return

    if isinstance(obj, Enum):
        obj = obj.value

    try:
        return obj.__dict__
    except AttributeError:
        pass

    if isinstance(obj, (float, int)):
        return obj
    else:
        return str(obj)


class StatusEnum(Enum):
    INITIALIZED = "INITIALIZED"

    RUNNING = "RUNNING"

    INTERRUPTED = "INTERRUPTED"

    FAILED = "FAILED"

    STOPPED = "STOPPED"

    # if the status FINISHED is received before stuff gehts starting, the skript should abort immediately
    FINISHED = "FINISHED"

    CUDA_OOM = "CUDA_OOM"

    # Gets set after last training epoch.
    # If TESTING is received before stuff gets started, skipp training and continue to test set
    TESTING = "TESTING"

    def __repr__(self):
        return self.name


@dataclass
class TrainJobStatus:
    # more or less statics
    # TODO maybe set them to be Path obj
    log_dir: str
    data_set: Optional[str]
    status_path: str
    wandb_id:str
    ckpt_dir: Optional[str] = None
    best_ckpt_path: Optional[str] = None
    latest_ckpt_path: Optional[str] = None

    # runtime
    hw_devices_used: Optional[List[str]] = field(default_factory=list)

    # current status
    status: StatusEnum = StatusEnum.INITIALIZED
    training_time: int = 0
    restarts: int = 0
    epochs: int = 0

    def save(self, path=None):
        try:
            if path is None:
                path = self.status_path
            log.debug(f'saving: {self}')
            with open(path, "w") as file:
                json.dump(self.__dict__, file, default=json_dumper, indent=2)
        except Exception as e:
            log.exception(f'!FAILED SAVING status! {self.status_path}', e)

    @staticmethod
    def load_status(path):
        # logger = log_control.spawn_new_logger(logger_name='status handler')
        with open(path, "r") as file:
            data = json.load(file)
        status: TrainJobStatus = dacite.from_dict(data_class=TrainJobStatus, data=data,
                                                  config=dacite.Config(cast=[Enum], strict_unions_match=True,
                                                                       strict=True))
        status.restarts += 1
        status.save()
        log.info('status loaded: %s', str(status))
        return status

    def set_best_ckpt_path(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        self.best_ckpt_path = path
        self.save()

    def set_ckpt_dir(self, path):
        """
        sets the ckpt path and saves the file
        dose not check if it exists!
        :param path:
        :return:
        """
        if not isinstance(path, Path):
            path= Path(path)
        self.ckpt_dir = path
        self.save()

    def get_ckpt_dir(self):
        """
        checks if the set ckpt path exists and returns it,
        None otherwise
        :return:
        """
        if self.ckpt_dir:
            if not isinstance(self.ckpt_dir, Path):
                self.ckpt_dir = Path(self.ckpt_dir)
            return self.ckpt_dir
        return None

    def set_latest_ckpt_path(self, path):
        if not isinstance(path, Path):
            path= Path(path)

        self.latest_ckpt_path = path
        self.save()

    def get_latest_ckpt_path(self):

        if self.latest_ckpt_path:
            if not isinstance(self.latest_ckpt_path, Path):
                self.latest_ckpt_path = Path(self.latest_ckpt_path)
            if self.latest_ckpt_path.exists():
                return self.latest_ckpt_path
        return None

    def epoch_finished(self):
        self.epochs += 1
        self.save()

    def set_running(self, running=True):
        self.status = StatusEnum.RUNNING if running else StatusEnum.STOPPED
        self.save()

    def set_testing(self):
        self.status = StatusEnum.TESTING
        self.save()

    def set_finished(self):
        self.status = StatusEnum.FINISHED
        self.save()

    def set_cudaoom(self):
        self.status = StatusEnum.CUDA_OOM
        self.save()

    def get_killed(self):
        self.status = StatusEnum.INTERRUPTED
        self.save()

    def add_hw_device(self, device):
        self.hw_devices_used.append(device)
        self.save()

    @contextmanager
    def training_active(self):
        self.set_running()
        start = time.time()

        try:
            yield None
        finally:
            end = time.time()
            self.training_time += int(end - start)
            if self.status == StatusEnum.RUNNING:
                self.set_running(False)
            self.save()
