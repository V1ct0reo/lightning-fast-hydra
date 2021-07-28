import unittest

import pandas as pd

from src.datamodules.datasets.window_makers.sliding_window_maker import MovementDataWindowMaker


class TestSequenceBasedMetric(unittest.TestCase):

    def test_unshuffled_window_start_idxs(self):
        data_list = []
        for p in range(3):
            for r in range(4):
                for t in range(6):
                    data_list.append([p, r, t, p * r * t, p * r * t * 2])
        dummy_data = pd.DataFrame(columns=['ParticipantID', 'repetition', 'time', 'f1', 'f2'],
                                  data=data_list)
        dummy_data['seq_id'] = dummy_data['ParticipantID']*100+dummy_data['repetition']
        window_maker = MovementDataWindowMaker(
            data=dummy_data, seq_identifier_col='seq_id', window_size=5, batch_size=8, data_is_sorted=False,
            labels_col='ParticipantID', sorting_cols=['ParticipantID', 'repetition', 'time'],
            feature_cols=list(dummy_data.columns), shuffle_windows=False
        )

        np.testing.Array


