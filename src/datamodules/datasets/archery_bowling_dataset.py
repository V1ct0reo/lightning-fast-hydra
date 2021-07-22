from pathlib import Path
from typing import List, Union

import pandas as pd
from torch.utils.data import Dataset


class ArcheryBowlingDataset(Dataset):
    """
    each dataset split is supposed to be instatiated with a propper dataframe
    """

    def __init__(self,
                 data: pd.DataFrame,
                 window_size: int,
                 batch_size: int,
                 name: str = None,
                 feature_cols: List[str] = None,
                 identifier_col: str = 'seq_id',
                 label_col: str = 'ParticipantID',
                 sorting_cols=None,
                 timestamp_col_for_sorting: str = None
                 ):

        if sorting_cols is None:
            sorting_cols = ['seq_id']
            if timestamp_col_for_sorting is not None:
                sorting_cols.append(timestamp_col_for_sorting)
        self.name = name
        if not feature_cols:
            feature_cols = ['CenterEyeAnchor_pos_X', 'CenterEyeAnchor_pos_Y', 'CenterEyeAnchor_pos_Z',
                            'LeftControllerAnchor_pos_X', 'LeftControllerAnchor_pos_Y', 'LeftControllerAnchor_pos_Z',
                            'RightControllerAnchor_pos_X', 'RightControllerAnchor_pos_Y', 'RightControllerAnchor_pos_Z']
        self.feature_cols = feature_cols
        self.identifier_col = identifier_col
        self.label_col = label_col
        columns = data.columns
        if not identifier_col in columns:
            # if not 'ScenarioID' in columns: # Scenario is not a feature and only 'archery' OR 'bowling' is in the same dataset
            #     # encode Scenario. 1 -> Archery; 2 -> Bowling
            #     data['ScenarioID'] = np.where(data['Scenario'] == 'Archery', 1, 2)
            # crate a unique sequenze identifier. ParticipantID and repetition might be bigger than 10.
            # TODO some kinda hash magic?
            data.loc[:, identifier_col] = (data['ParticipantID'] * 100
                                           + data['repetition']
                                           # + data['ScenarioID'] * 10 # scenarioID could be ignored, since we only look at on scenario at a time
                                           # + data['study_session']  # same for session ... a dataset only contains session 1 OR 2
                                           )

        # make ParicipantsID zero-indexed (original starts at 1)
        data['ParticipantID'] = data['ParticipantID'] - 1
        self.data = data
        self.batch_size = batch_size
        self.window_size = window_size
        self.identifier_col = identifier_col
        self.window_maker = MovementDataWindowMaker(
            data=data, seq_identifier_col=identifier_col, window_size=window_size, batch_size=batch_size,
            sorting_cols=sorting_cols, feature_cols=feature_cols, labels_col=label_col, data_is_sorted=False)

    def __getitem__(self, batch_index) -> T_co:
        """
        get a complete tensor of a data-batch. Shape should be (batch_size,window_size,num_features)
        :param batch_index:
        :return: (x,Y); where x is a tensor of size (batch_size,window_size,num_features) and Y is a tensor of shape(batch_size)
        """
        return self.window_maker.get_batch_from_idx(batch_index, self.batch_size)

    def __len__(self):
        return self.window_maker.total_num_batches

    def __repr__(self):
        return f"Dataset({self.name!r}, batch_size={self.batch_size!r}, window_size={self.window_size!r}, " \
               f"num_frames={self.window_maker.num_entries!r}, num_windows={self.window_maker.total_num_windows!r}, " \
               f"num_batches={self.window_maker.total_num_batches!r})"

    def __rich_repr__(self):
        yield self.name
        yield "batch_size", self.batch_size
        yield "window_size", self.window_size
        yield "num_frames", self.window_maker.num_entries
        yield "num_windows", self.window_maker.total_num_windows
        yield "num_batches", self.window_maker.total_num_batches

    @staticmethod
    def create_from_files(file_list: [Path], window_size,
                          batch_size,
                          name=None, feature_cols: List[str] = None,
                          identifier_col: str = 'seq_id',
                          label_col: str = 'ParticipantID',
                          sorting_cols: List[str] = None,
                          ):
        """
        instantiates Dataset Objects
        :param name:
        :param feature_cols:
        :param identifier_col:
        :param sorting_cols:
        :param label_col:
        :param file_list: list of Path's to csv files to be considered for this dataset.
        :param window_size:
        :param batch_size:
        :return: ArcheryBowlingDataset
        """
        all_data = ArcheryBowlingDataModule.load_dataframe_from_multiple_files(file_list)
        return ArcheryBowlingDataset(all_data, window_size, batch_size, name=name, feature_cols=feature_cols,
                                     identifier_col=identifier_col, label_col=label_col, sorting_cols=sorting_cols)

    @staticmethod
    def create_from_dataframe(data: Union[pd.DataFrame, Path], window_size, batch_size,
                              name=None, feature_cols: List[str] = None,
                              identifier_col: str = 'seq_id',
                              label_col: str = 'ParticipantID',
                              sorting_cols: List[str] = None,
                              ):
        """ given a pd.Dataframe or path to a single csv, instantiate a Dataset Object of it"""
        if isinstance(data, Path):
            data = pd.load_csv(data)
        return ArcheryBowlingDataset(data, window_size, batch_size, name=name, feature_cols=feature_cols,
                                     identifier_col=identifier_col, label_col=label_col, sorting_cols=sorting_cols)
