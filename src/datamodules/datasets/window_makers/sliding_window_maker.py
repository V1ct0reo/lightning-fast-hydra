# TODO add shuffle option
import math

import numpy as np
import pandas as pd
import torch


class MovementDataWindowMaker:
    def __init__(self, data: pd.DataFrame, seq_identifier_col: str, window_size: int,
                 batch_size: int = 1, data_is_sorted=False, labels_col: str = 'ParticipantID',
                 sorting_cols: [str] = None, feature_cols: [str] = None ):

        if not data_is_sorted:
            if sorting_cols is None or len(sorting_cols) < 1:
                raise AttributeError('if input data is not sorted already, sorting_cols need to be specified')
            data = data.sort_values(by=sorting_cols)

        self.y = data[labels_col].values

        if feature_cols is None or len(feature_cols) < 1:
            self.feature_cols = list(data.columns)
            self.num_features = len(self.feature_cols)
        else:
            self.num_features = len(feature_cols)
            self.feature_cols = feature_cols

        allowed_dtypes = ['float64', 'float32', 'float16', 'complex64', 'complex128', 'int64', 'int32', 'int16', 'int8',
                          'uint8', 'bool']
        illegal_dtypes_found = [x for x in list(data[self.feature_cols].dtypes) if x not in allowed_dtypes]

        if len(illegal_dtypes_found) > 0:
            raise AttributeError( # TODO catch error higher up
                f'DataFrame contains an illegal dtype:{[x for x in illegal_dtypes_found]}. Allowed are only {allowed_dtypes}')

        self.labels_col = labels_col
        self.num_entries = len(data)
        self.window_size = window_size
        self.sequenz_start_idxs: np.ndarray = self._compute_sequenz_start_ids(
            data[seq_identifier_col].values.astype(int))
        self.window_pointer: int = 0
        self.sequenz_lens: np.ndarray = np.diff(np.append(self.sequenz_start_idxs, self.num_entries)).astype(int)
        self.batch_size = batch_size
        self.data = data[self.feature_cols]

        if np.any(self.sequenz_lens < self.window_size):
            raise AttributeError(f'window size ({window_size}) musst be smaller than shortest sequence!')

        # create a list of indices, that might be used as window_Start_id

        # holding the entire array simply to do computations once vectorized instead of once for each sequence
        self.last_window_start_idxs_per_sequenz = self.sequenz_lens - window_size + self.sequenz_start_idxs
        window_start_idxs = []
        for seq, idx in enumerate(self.sequenz_start_idxs):
            # TODO is this a dully implementation?
            #  there has to be a smart hyper-complex np index slicing to accomplish the same
            window_start_idxs += range(idx, self.last_window_start_idxs_per_sequenz[seq] + 1)
        self.window_start_idxs = np.array(window_start_idxs)  # EXCLUDE all idx, which would not satisfy a full array

        # TODO maybe shuffle window_start_idxs, since from now on, it dosnt matter, which idx is used for a window.
        #  maybe maybe consider batch balancing though?

        self.total_num_windows = self.window_start_idxs.size
        # if the last batch surpasses the end of data, it will be filled up from the start of data again
        self.total_num_batches = math.ceil(self.total_num_windows / batch_size)

    # @deprecate
    def get_next_batch_idxs(self, batch_size: int):

        batch_idxs = np.empty((batch_size, 2), dtype=np.uintp)
        if (self.window_pointer + batch_size) >= len(self.window_start_idxs):
            windows_available = self.total_num_windows - self.window_pointer
            batch_idxs[:windows_available, 0] = self.window_start_idxs[
                                                self.window_pointer:self.window_pointer + windows_available]
            self.window_pointer = 0
            batch_idxs[windows_available:, 0] = self.window_start_idxs[
                                                self.window_pointer:self.window_pointer + batch_size - windows_available]
            batch_idxs[:, 1] = batch_idxs[:, 0] + self.window_size
            self.window_pointer = batch_size - windows_available
        else:
            batch_idxs[:, 0] = self.window_start_idxs[self.window_pointer:self.window_pointer + batch_size]
            batch_idxs[:, 1] = batch_idxs[:, 0] + self.window_size
            self.window_pointer += batch_size
        return batch_idxs

    # @deprecate
    def get_next_batch_data(self, batch_size):
        """internally calls get_next_batch_idxs, which increments window_pointer

        returns a np.ndarray of shape (batch size , window size , num features)
        basically a ready to go batch of sequenz windows from a dataframe
        """
        batch = np.empty((batch_size, self.window_size, self.num_features), dtype='float32')
        for i, (start, end) in enumerate(self.get_next_batch_idxs(batch_size)):
            batch[i] = self.data.iloc[start:end]
        # batch = np.concatenate(
        #     [self.data.iloc[start:end] for start, end in self.get_next_batch_idxs(batch_size)]
        # ).reshape((batch_size, self.window_size, self.num_features))
        return batch

    def get_batch_idxs_from_idx(self, idx, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        idx = idx * batch_size
        batch_idxs = np.empty((batch_size, 2), dtype=np.uintp)
        if (idx + batch_size) >= len(self.window_start_idxs):
            windows_available = self.total_num_windows - idx
            batch_idxs[:windows_available, 0] = self.window_start_idxs[
                                                idx:idx + windows_available]
            idx = 0
            batch_idxs[windows_available:, 0] = self.window_start_idxs[
                                                idx:idx + batch_size - windows_available]
            batch_idxs[:, 1] = batch_idxs[:, 0] + self.window_size

        else:
            batch_idxs[:, 0] = self.window_start_idxs[idx:idx + batch_size]
            batch_idxs[:, 1] = batch_idxs[:, 0] + self.window_size

        return batch_idxs

    def get_batch_from_idx(self, idx, batch_size=None):
        """ used by Dataset class. apparently it asks for a specific data_index inside __getitem__"""
        # TODO pre allocate an empty array instead of concatenating...
        if not batch_size:
            batch_size = self.batch_size
        slices = self.get_batch_idxs_from_idx(idx, batch_size)
        batch = np.concatenate(
            [self.data.iloc[start:end] for start, end in
             iter(slices)]
        ).reshape((batch_size, self.window_size, self.num_features))

        return torch.from_numpy(batch).float(), torch.from_numpy(self.y[slices.T[0]])

    @staticmethod
    def _compute_sequenz_start_ids(data: np.ndarray):
        """expects a 1D array and return an array of indexes. At each index, a new sequenz starts"""
        return np.where(np.diff(data, prepend=np.nan))[0]