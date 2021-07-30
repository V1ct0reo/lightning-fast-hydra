from pathlib import Path
from typing import Union, List, Dict, Optional

import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import transforms

from src.utils.utils import get_logger


class ArcheryBowlingDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_root: str,
                 test: bool = False,
                 val_ratio: float = None,
                 batch_size: int = 1,
                 window_size: int = 10,
                 normalisation: str = 'WithoutNormalization',
                 szenario: str = 'Archery',
                 features: List[str] = ['CenterEyeAnchor_pos_X', 'LeftVirtualHand_pos_X', 'RightVirtualHand_pos_X'],
                 identifier_col: str = 'seq_id',
                 label_col: str = 'ParticipantID',
                 sorting_cols: List[str] = None,
                 num_workers: int = 1,
                 shuffle_windows=False
                 ):
        super(ArcheryBowlingDataModule, self).__init__()
        self.num_workers = num_workers
        self.logger = get_logger(name='A-B-DataModule')
        self.szenario = szenario
        self.features = features
        self.identifier_col = identifier_col if identifier_col is not None else 'seq_id'
        self.label_col = label_col if label_col is not None else 'ParticipantID'
        self.sorting_cols = sorting_cols
        self.normalisation = normalisation
        self.window_size = window_size
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.separate = test
        self.data_root = Path(data_root)  # Path is just more convenient
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.shuffle_windows = shuffle_windows
        self.num_features = len(features)
        self.dims = (self.window_size, self.num_features)
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.logger.info('__init__ done.')

    def setup(self, stage: Optional[str] = None) -> None:
        # do i want to load all data at once, and slice afterwords?
        # slice all modulo matching repetitions to validation
        # keep the remaining as train
        # drop unused columns
        # initiate DatasetObjects and return them
        # return ArcheryBowlingDataset(None, 1, 1), ArcheryBowlingDataset(None, 1, 1), ArcheryBowlingDataset(None, 1, 1)

        if stage in (None, 'fit'):  # TODO no validation set throws a Nonetype Error on val data loader...
            self.logger.info(f'stage:{stage}. creating Dataset...')
            # regexing or sorting the file path seems to be a pain. therefore ill load all relevant (normalized + session1)
            train_val_files = self.get_file_list(session=1)
            train_val_files = list(train_val_files)
            self.logger.info(f'found {len(train_val_files)} files.')
            train_val_df = ArcheryBowlingDataModule.load_dataframe_from_multiple_files(train_val_files)

            # TODO refactor this ifelse structure to a neat structure
            if self.val_ratio and self.val_ratio > 0:  # not none and > 0
                modulo = int(1 / self.val_ratio)
                if modulo > 12 or modulo < 2:
                    self.logger.info(
                        f'validation split ratio({self.val_ratio}) was set, '
                        f'but would result in either all or no data being available for training. '
                        f'Therefore all Data will be used as train-set!')
                    from src.datamodules.datasets.archery_bowling_dataset import ArcheryBowlingDataset
                    self.train_dataset = ArcheryBowlingDataset.create_from_dataframe(train_val_df, self.window_size,
                                                                                     self.batch_size, name='TRAIN',
                                                                                     feature_cols=self.features,
                                                                                     identifier_col=self.identifier_col,
                                                                                     label_col=self.label_col,
                                                                                     shuffle_windows=self.shuffle_windows,
                                                                                     sorting_cols=self.sorting_cols
                                                                                     )
                else:
                    val_df = train_val_df[train_val_df['repetition'] % modulo == 0]
                    from src.datamodules.datasets.archery_bowling_dataset import ArcheryBowlingDataset
                    self.val_dataset = ArcheryBowlingDataset.create_from_dataframe(val_df, self.window_size,
                                                                                   self.batch_size, name='VAL',
                                                                                   feature_cols=self.features,
                                                                                   identifier_col=self.identifier_col,
                                                                                   label_col=self.label_col,
                                                                                   shuffle_windows=self.shuffle_windows,
                                                                                   sorting_cols=self.sorting_cols
                                                                                   )
                    del val_df

                    train_df = train_val_df[train_val_df['repetition'] % modulo != 0]
                    del train_val_df
                    self.train_dataset = ArcheryBowlingDataset.create_from_dataframe(train_df, self.window_size,
                                                                                     self.batch_size, name='TRAIN',
                                                                                     feature_cols=self.features,
                                                                                     identifier_col=self.identifier_col,
                                                                                     label_col=self.label_col,
                                                                                     shuffle_windows=self.shuffle_windows,
                                                                                     sorting_cols=self.sorting_cols
                                                                                     )
                    del train_df
            else:
                from src.datamodules.datasets.archery_bowling_dataset import ArcheryBowlingDataset
                self.train_dataset = ArcheryBowlingDataset.create_from_dataframe(train_val_df, self.window_size,
                                                                                 self.batch_size, name='TRAIN',
                                                                                 feature_cols=self.features,
                                                                                 identifier_col=self.identifier_col,
                                                                                 label_col=self.label_col,
                                                                                 shuffle_windows=self.shuffle_windows,
                                                                                 sorting_cols=self.sorting_cols
                                                                                 )
                self.val_dataset = None

            self.logger.info('train/val Data initialized!')

        if stage in (None, 'test'):
            # slice all 'session2' entries for test data
            # create a list of paths for test data files (basically everything with session 2
            self.logger.info(f'stage:{stage}. creating Dataset...')
            test_files = self.get_file_list(session=2)
            test_files = (list(test_files))
            self.logger.info(f'found {len(test_files)} test-files.')

            # create test Dataset
            from src.datamodules.datasets.archery_bowling_dataset import ArcheryBowlingDataset
            test_df = ArcheryBowlingDataModule.load_dataframe_from_multiple_files(test_files)
            computed_batch_size = self.batch_size
            rest = len(test_df) % self.batch_size
            computed_batch_size -= rest
            self.test_dataset = ArcheryBowlingDataset.create_from_dataframe(test_df, self.window_size, computed_batch_size,
                                                                            name='TEST', feature_cols=self.features,
                                                                            identifier_col=self.identifier_col,
                                                                            label_col=self.label_col,
                                                                            shuffle_windows=False,
                                                                            sorting_cols=self.sorting_cols
                                                                            )
            self.logger.info('test Data initialized!')

        self.logger.info(f'Datasets are setup.')
        self.logger.info(self)

    def get_file_list(self, session=1):
        train_val_files = self.data_root.glob(f'{self.szenario}*{self.normalisation}*session{session}*.csv')
        return train_val_files

    @staticmethod
    def load_dataframe_from_multiple_files(file_list: List[Path]):
        df_list = []
        for i in file_list:
            tmp = pd.read_csv(i)
            df_list.append(tmp)
        return pd.concat(df_list, ignore_index=True)

    def _create_info_dict(self):
        return {
            'train dataset': None if not self.train_dataset else str(self.train_dataset),
            'val dataset': None if not self.val_dataset else str(self.val_dataset),
            'test dataset': None if not self.test_dataset else str(self.test_dataset),
            'dims': self.dims,
            '#batches': len(self.test_dataset),
            'window size': self.window_size,
            'batch size': self.batch_size,
            'normalisation name': self.normalisation
        }

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers
                          )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=None, num_workers=self.num_workers
                          )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # TODO handle num_workers...
        return DataLoader(self.test_dataset, batch_size=None, num_workers=self.num_workers
                          )

    def __repr__(self):
        return f"DataModule(train_dataset={self.train_dataset!r}, " \
               f"val_dataset={self.val_dataset!r}, " \
               f"test_dataset={self.test_dataset!r}, " \
               f"dims={self.dims!r}, " \
               f"normalisation_name={self.normalisation!r}), " \
               f"Szenario={self.szenario})"

    def __rich_repr__(self):
        yield "train_dataset", self.train_dataset
        yield "val_dataset", self.val_dataset
        yield "test_dataset", self.test_dataset
        yield "dims", self.dims
        yield "normalisation_name", self.normalisation
        yield "szenario", self.szenario
