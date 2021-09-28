import collections
import os
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow_hub
import wandb
import seaborn as sn

import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
from omegaconf import DictConfig

from src.utils import utils

log = utils.get_logger(__name__)


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def tf_train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    potential_subdir = Path(os.getcwd()).name
    if not potential_subdir == config.run_name:
        config.run_name = str(config.run_name) + '/' + Path(os.getcwd()).name

    MODEL_HANDLES= {
        'MobileNet_v2': 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
        'EfficientNet_b0': 'https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1',
        'MobileNet_v3:': 'https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5',
        'ResNet_v2_50': 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5'
    }

    # wandb.tensorboard.path(root_logdir='tensorboard/')
    wandb.login(key=os.environ['WANDB_API_KEY'])
    try:
        tf.keras.backend.clear_session()
        # DATA PREPROCESSING
        if config.ONLY_LOCATIONS:
            config.data_dir = str(config.data_dir).replace('Lextion-Images','Lexion-Locations')
            config.N_CLASSES = 10


        flat_config = flatten(config, sep='.')
        DATA_DIR = Path(config.data_dir)
        wandb.init(config=flat_config,
                   project=config.project_id,
                   name=config.run_name,
                   job_type='train',
                   sync_tensorboard=True)

        train_dir = DATA_DIR.joinpath('train')
        train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=config.batch_size,
                                                     image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))

        val_dir = DATA_DIR.joinpath('val')
        val_dataset = image_dataset_from_directory(val_dir, shuffle=True, batch_size=config.batch_size,
                                                   image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))

        test_dir = DATA_DIR.joinpath('test')
        test_dataset = image_dataset_from_directory(test_dir, shuffle=False, batch_size=config.batch_size,
                                                    image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))

        # TODO augemtations

        # TODO check pixel space [0:1]? (-1:1]? ...? -> match with model used..
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


        keras_model = Sequential([
            tf.keras.layers.InputLayer(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)),
            tensorflow_hub.KerasLayer(MODEL_HANDLES[config.BASE_MODEL], trainable=False, name=config.BASE_MODEL),
            tf.keras.layers.Dropout(rate=config.DROPOUT),
            tf.keras.layers.Dense(config.N_CLASSES, activation='softmax')
        ])
        keras_model.build((None, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
        keras_model.summary()
        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=config.LEARN_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc']
        )
        # -- INITIAL TRAIN ----
        if config.INITIAL_EPOCHS > 0:
            log.info(f'starting Initial Training for {config.INITIAL_EPOCHS} Epochs..')
            _ = keras_model.fit(train_dataset, validation_data=val_dataset, epochs=config.INITIAL_EPOCHS)


        # --- REAL TRAIN --
        log.info(f'starting real Training for {config.FINE_TUNE_EPOCHS} Epochs..')
        keras_model.layers[0].trainable=True
        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=config.LEARN_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc']
        )
        history = keras_model.fit(train_dataset, validation_data=val_dataset, epochs=config.FINE_TUNE_EPOCHS,
                                  callbacks=[
                                      tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/',
                                                                         monitor='val_acc',
                                                                         save_best_only=True),
                                      wandb.keras.WandbCallback(monitor='val_acc', mode='max', save_model=True,),
                                      tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience=15)
                                  ])

        # --TEST RUN ---
        test_predictions = []
        test_targets = []
        for (x, y) in test_dataset:
            test_predictions.append(np.argmax(keras_model.predict(x), axis=-1))
            test_targets.append(y.numpy())
        test_predictions = np.concatenate(test_predictions)
        test_targets = np.concatenate(test_targets)
        confusion_matrix = metrics.confusion_matrix(y_true=test_targets, y_pred=test_predictions, normalize='true')
        # set figure size
        plt.figure(figsize=(14, 8))
        # set labels size
        sn.set(font_scale=1.4)
        # set font size
        sn.heatmap(np.round(confusion_matrix,decimals=3), annot=True, annot_kws={"size": 8}, fmt="g",
                   xticklabels=test_dataset.class_names, yticklabels=test_dataset.class_names)
        # names should be uniqe or else charts from different experiments in wandb will overlap
        plt.tight_layout()
        wandb.log({f"test/confusion_matrix/{config.run_name}": wandb.Image(plt)}, commit=False)
        # reset plot
        plt.clf()

        acc = tf.keras.metrics.Accuracy()
        acc.update_state(test_targets, test_predictions)
        # tf.keras.metrics.categorical_accuracy(test_targets,test_predictions)
        wandb.log({'test_acc': acc.result()})
        log.info(f'TRAINING DONE FOR {config.BASE_MODEL}\n')
    except Exception as e:
        log.exception(e)
    finally:
        log.info(f'FINISHING WANDB {config.BASE_MODEL}\n')
        wandb.finish()
    # TODO:
    #   finetuning process: start with a frozen basemodel and train the classification head
    #   THEN unfreeze a couple of layers and train AGAIN

    # # Return metric score for hyperparameter optimization
    # optimized_metric = config.get("optimized_metric")
    # if optimized_metric:
    #     return trainer.callback_metrics[optimized_metric]
