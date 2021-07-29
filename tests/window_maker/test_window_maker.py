import unittest
import numpy as np
import pandas as pd

from src.datamodules.datasets.window_makers.sliding_window_maker import MovementDataWindowMaker


class TestMovementDataWindowMakerTest(unittest.TestCase):

    # def setUp(self):
    #     self.df = pd.read_csv(root_path.CODE_ROOT.joinpath('movement_testing', 'example.csv'), delimiter=';')
    def setUp(self) -> None:
        self.small_df = pd.read_csv('S:\\Studium_nosync\\lightning-fast-hydra\\tests\\window_maker\\example.csv',
                                    delimiter=';').drop(columns=['strCol', 'strId'])
        self.bigger_df = self.small_df.append(pd.DataFrame(data=[
            [3, 'her', 400, 410, 420, 'fdsa'],
            [3, 'her', 401, 411, 421, 'fdsa'],
            [3, 'her', 402, 412, 422, 'fdsa'],
            [3, 'her', 403, 413, 423, 'fdsa'],
            [3, 'her', 404, 414, 424, 'fdsa'],
            [3, 'her', 405, 415, 425, 'fdsa'],
            [4, 'her', 501, 511, 521, 'fdsa'],
            [4, 'her', 502, 512, 522, 'fdsa'],
            [4, 'her', 503, 513, 523, 'fdsa'],
            [4, 'her', 504, 514, 524, 'fdsa'],
            [4, 'her', 505, 515, 525, 'fdsa']
        ], columns=["id", "strId", "feature0", "feature1", "feature2", "strCol"]), ignore_index=True)
        self.bigger_df = self.bigger_df.drop(columns=["strId", "strCol"])
        self.million_df = None

    def _generate_int_df(self):
        if not self.million_df:
            num_entries = 1_000_000
            self.million_df = pd.DataFrame(
                [(id, i + 1, i * 2, i + 1) for id, i in
                 zip(np.linspace(0, 20, num_entries, endpoint=False), range(num_entries))],
                columns=['id', 'f1', 'f2', 'intId']
            )
        return self.million_df

    def test_state_simple_init(self):
        df = self.small_df
        id = 'id'
        window_size = 2
        data_is_sorted = True
        sorting_cols = None

        ex_num_entries = 12
        ex_window_size = 2
        ex_seq_start_idxs = np.array([0, 4, 8])
        ex_seq_lens = np.array([4, 4, 4])
        ex_last_window_idxs = np.array([2, 6, 10])
        ex_window_start_idxs = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10])
        ex_total_windows = 9
        ex_total_batches = 9

        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col=id,
            labels_col='intId',
            window_size=window_size,
            data_is_sorted=data_is_sorted,
            sorting_cols=sorting_cols
        )

        self.WindowMakerState(ex_last_window_idxs, ex_num_entries, ex_seq_lens, ex_seq_start_idxs, ex_total_windows,
                              ex_window_size, ex_window_start_idxs, ex_total_batches, mdwm)

    def test_state_odd_seq_length(self):
        df = self.bigger_df
        id = 'id'

        window_size = 2
        data_is_sorted = True
        sorting_cols = None

        ex_num_entries = 12 + 11
        ex_window_size = 2
        ex_seq_start_idxs = np.array([0, 4, 8, 12, 18])
        ex_seq_lens = np.array([4, 4, 4, 6, 5])
        ex_last_window_idxs = np.array([2, 6, 10, 16, 21])
        ex_window_start_idxs = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21])
        ex_total_windows = ex_window_start_idxs.size
        ex_total_batches = ex_window_start_idxs.size

        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col=id, labels_col='intId',

            window_size=window_size,
            data_is_sorted=data_is_sorted,
            sorting_cols=sorting_cols
        )
        self.WindowMakerState(ex_last_window_idxs, ex_num_entries, ex_seq_lens, ex_seq_start_idxs, ex_total_windows,
                              ex_window_size, ex_window_start_idxs, ex_total_batches, mdwm)

    def test_next_batch_idx_generation_twice(self):
        df = self.bigger_df
        id = 'id'

        window_size = 2
        data_is_sorted = True
        sorting_cols = None

        batch_size = 5

        ex_batch = np.array([
            (0, 2),
            (1, 3),
            (2, 4),
            (4, 6),
            (5, 7)
        ])

        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col=id, labels_col='intId',

            window_size=window_size,
            data_is_sorted=data_is_sorted,
            sorting_cols=sorting_cols
        )

        batch = mdwm.get_next_batch_idxs(batch_size)

        self.assertEqual(batch.shape, ex_batch.shape)
        np.testing.assert_array_equal(batch, ex_batch)
        self.assertEqual(mdwm.window_pointer, 5)

        ex_batch = np.array([
            (6, 8),
            (8, 10),
            (9, 11),
            (10, 12),
            (12, 14)
        ])

        batch = mdwm.get_next_batch_idxs(batch_size)
        self.assertEqual(batch.shape, ex_batch.shape)
        np.testing.assert_array_equal(batch, ex_batch)
        self.assertEqual(mdwm.window_pointer, 10)

    def test_total_batches(self):
        df = self.small_df
        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col='id', labels_col='intId',
            batch_size=3,
            window_size=3,
            data_is_sorted=True
        )
        self.assertEqual(mdwm.total_num_batches, 2)

    def test_total_batches_loop_back_up(self):
        df = self.small_df
        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col='id', labels_col='intId',
            batch_size=4,
            window_size=3,
            data_is_sorted=True
        )
        self.assertEqual(mdwm.total_num_batches, 2)

    def test_get_full_next_batch(self):
        df = self.bigger_df
        id = 'id'
        window_size = 2
        data_is_sorted = True
        sorting_cols = None

        batch_size = 3

        ex_batch_slices = np.array([
            (0, 2),
            (1, 3),
            (2, 4),
        ])
        ex_batch = np.array([
            [[0, 100, 110, 120],
             [0, 101, 111, 121]],

            [[0, 101, 111, 121],
             [0, 102, 112, 122]],

            [[0, 102, 112, 122],
             [0, 103, 113, 123]]
        ])

        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col=id, labels_col='intId',
            feature_cols=['id', 'feature0', 'feature1', 'feature2'],

            window_size=window_size,
            data_is_sorted=data_is_sorted,
            sorting_cols=sorting_cols
        )

        batch = mdwm.get_next_batch_data(batch_size=batch_size)
        np.testing.assert_array_equal(batch, ex_batch)
        self.assertEqual(mdwm.window_pointer, batch_size)

    def test_get_next_batch_idxs_by_idx_0(self):
        df = self.small_df
        window_size = 2
        data_is_sorted = True
        sorting_cols = None
        batch_size = 3

        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col='id', labels_col='intId',

            window_size=window_size,
            data_is_sorted=data_is_sorted,
            sorting_cols=sorting_cols
        )

        batch_idxs = mdwm.get_batch_idxs_from_idx(0, batch_size)
        ex_batch_idxs = np.array([
            [0, 2],
            [1, 3],
            [2, 4]
        ])
        np.testing.assert_array_equal(batch_idxs, ex_batch_idxs)

    def test_get_next_batch_idxs_by_idx_2(self):
        df = self.small_df
        window_size = 2
        data_is_sorted = True
        sorting_cols = None
        batch_size = 3

        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col='id', labels_col='intId',

            window_size=window_size,
            data_is_sorted=data_is_sorted,
            sorting_cols=sorting_cols
        )

        batch_idxs = mdwm.get_batch_idxs_from_idx(2, batch_size)
        ex_batch_idxs = np.array([
            [8, 10],
            [9, 11],
            [10, 12]
        ])
        np.testing.assert_array_equal(batch_idxs, ex_batch_idxs)

    def test_z_million_entries_next_batch_idxs(self):
        df = self._generate_int_df()
        data_is_sorted = True
        sorting_cols = None
        id = 'id'
        window_size = 20
        batch_size = 50_000
        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col=id,
            window_size=window_size, labels_col='intId',

            data_is_sorted=data_is_sorted,
            sorting_cols=sorting_cols
        )

        batch = mdwm.get_next_batch_idxs(batch_size)
        self.assertEqual(batch.shape, (batch_size, 2))

    def test_z_million_entries_full_next_batch(self):
        df = self._generate_int_df()
        data_is_sorted = True
        sorting_cols = None
        id = 'id'
        window_size = 2
        batch_size = 10_000
        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col=id, labels_col='intId',
            feature_cols=['f1', 'f2', 'intId'],
            window_size=window_size,
            data_is_sorted=data_is_sorted,
            sorting_cols=sorting_cols
        )

        batch = mdwm.get_next_batch_idxs(batch_size)
        self.assertEqual(batch.shape, (batch_size, 2))

        batch = mdwm.get_next_batch_data(batch_size)
        self.assertEqual(batch.shape, (batch_size, window_size, 3))

    def test_z_too_big_window_AttributeError(self):
        df = self._generate_int_df()
        data_is_sorted = True
        sorting_cols = None
        id = 'id'
        window_size = 500_000
        batch_size = 10_000

        with self.assertRaises(AttributeError) as context:
            MovementDataWindowMaker(
                data=df,
                seq_identifier_col=id, labels_col='intId',

                window_size=window_size,
                data_is_sorted=data_is_sorted,
                sorting_cols=sorting_cols
            )

    def test_z_big_window_long_data_next_batch(self):
        df = self._generate_int_df()
        data_is_sorted = True
        sorting_cols = None
        id = 'id'
        window_size = 5000
        batch_size = 1000

        mdwm = MovementDataWindowMaker(
            data=df,
            seq_identifier_col=id, labels_col='intId',
            feature_cols=['f1', 'f2', 'intId'],
            window_size=window_size,
            data_is_sorted=data_is_sorted,
            sorting_cols=sorting_cols
        )

        batch = mdwm.get_next_batch_idxs(batch_size)
        self.assertEqual(batch.shape, (batch_size, 2))

        batch = mdwm.get_next_batch_data(batch_size)
        self.assertEqual(batch.shape, (batch_size, window_size, 3))

    def test_simple_batch_idx_from_idx(self):
        df = self.bigger_df
        id = 'id'

        window_size = 2
        data_is_sorted = True
        sorting_cols = None

        batch_size = 5

        ex_batch = np.array([
            (0, 2),
            (1, 3),
            (2, 4),
            (4, 6),
            (5, 7)
        ])

        mdwm = MovementDataWindowMaker(
            labels_col='intId',
            batch_size=3,
            window_size=2,
            data_is_sorted=True,
            data=self.small_df,
            seq_identifier_col='id'
        )
        batch_inx = 0

        batch = mdwm.get_batch_idxs_from_idx(batch_inx, batch_size)
        np.testing.assert_array_equal(batch, ex_batch)

    def test_simple_batch_idx_from_idx_1(self):
        df = self.small_df
        id = 'id'

        batch_size = 3

        ex_batch = np.array([
            (4, 6),
            (5, 7),
            (6, 8)
        ])

        mdwm = MovementDataWindowMaker(
            labels_col='intId',
            batch_size=3,
            window_size=2,
            data_is_sorted=True,
            data=df,
            seq_identifier_col='id'
        )
        batch_inx = 1

        batch = mdwm.get_batch_idxs_from_idx(batch_inx, batch_size)
        np.testing.assert_array_equal(batch, ex_batch)

    def test_simple_batch_idx_from_idx_2(self):
        df = self.bigger_df
        id = 'id'

        ex_batch = np.array([
            (12, 15),
            (13, 16),
            (14, 17)
        ])

        mdwm = MovementDataWindowMaker(
            labels_col='intId',
            batch_size=3,
            window_size=3,
            data_is_sorted=True,
            data=df,
            seq_identifier_col='id'
        )
        batch_inx = 2

        batch = mdwm.get_batch_idxs_from_idx(batch_inx, 3)
        np.testing.assert_array_equal(batch, ex_batch)

    def test_simple_batch_from_idx(self):
        id = 'id'
        df = pd.DataFrame(data=[
            [3, 'her', 400, 410, 420, 'fdsa', 4],
            [3, 'her', 401, 411, 421, 'fdsa', 4],
            [3, 'her', 402, 412, 422, 'fdsa', 4],
            [3, 'her', 403, 413, 423, 'fdsa', 4],
            [3, 'her', 404, 414, 424, 'fdsa', 4],
            [3, 'her', 400, 410, 420, 'fdsa', 4],
            [4, 'her', 501, 511, 521, 'fdsa', 5],
            [4, 'her', 502, 512, 522, 'fdsa', 5],
            [4, 'her', 503, 513, 523, 'fdsa', 5],
            [4, 'her', 504, 514, 524, 'fdsa', 5],
            [4, 'her', 504, 514, 524, 'fdsa', 5]
        ], columns=["id", "strId", "feature0", "feature1", "feature2", "strCol", "intId"])
        df = df.drop(columns=["strCol", "strId"])
        ex_batch = np.array([
            [[3, 400, 410, 420],
             [3, 401, 411, 421],
             [3, 402, 412, 422]],
            [[3, 401, 411, 421],
             [3, 402, 412, 422],
             [3, 403, 413, 423]],
            [[3, 402, 412, 422],
             [3, 403, 413, 423],
             [3, 404, 414, 424]]
        ])

        ex_y = np.array([
            4, 4, 4
        ])

        mdwm = MovementDataWindowMaker(
            labels_col='intId',
            batch_size=3,
            window_size=3,
            data_is_sorted=True,
            data=df,
            seq_identifier_col='id',
            feature_cols=['id', 'feature0', 'feature1', 'feature2']

        )
        batch_inx = 0

        batch, y, _ = mdwm.get_batch_from_idx(batch_inx, 3)
        np.testing.assert_array_equal(batch.numpy(), ex_batch)
        np.testing.assert_array_equal(y.numpy(), ex_y)

    def test_string_col_in_df(self):
        df = pd.DataFrame([[1, 'fdsa', 1.], [1, 'fdsa', 1.]], columns=['a', 'b', 'c'])
        with self.assertRaises(AttributeError):
            mdwm = MovementDataWindowMaker(
                labels_col='a',
                batch_size=1,
                window_size=1,
                data_is_sorted=True,
                data=df,
                seq_identifier_col='a'
            )

    def test_batch_from_idx_change_in_y(self):
        df = self.bigger_df
        mdwm = MovementDataWindowMaker(
            labels_col='intId',
            batch_size=3,
            window_size=3,
            data_is_sorted=True,
            data=df,
            seq_identifier_col='id',
            feature_cols=['id', 'feature0', 'feature1', 'feature2']
        )
        ex_batch = np.array([
            [[0, 100, 110, 120],
             [0, 101, 111, 121],
             [0, 102, 112, 122]],
            [[0, 101, 111, 121],
             [0, 102, 112, 122],
             [0, 103, 113, 123]],
            [[1, 200, 210, 220],
             [1, 201, 211, 221],
             [1, 202, 212, 222]]
        ])
        ex_y = np.array([
            1, 1, 2
        ])

        batch, y, _ = mdwm.get_batch_from_idx(0, 3)
        np.testing.assert_array_equal(batch, ex_batch, 'did not get the expected batch data')
        np.testing.assert_array_equal(y, ex_y, 'did not get the expected labels')

    def test_some_singe_entry_batch_with_window_size_3_with_sorting_single_col(self):
        shuffled_df = self.bigger_df
        shuffled_df['time'] = list(range(4)) * 3 + list(range(6)) + list(range(5))
        shuffled_df = shuffled_df.sample(frac=1).reset_index(drop=True)
        mdwm = MovementDataWindowMaker(
            data=shuffled_df,
            seq_identifier_col='id',
            labels_col='intId',
            window_size=3,
            data_is_sorted=False,
            sorting_cols=['id', 'time'],
            batch_size=1
        )

        actual_5_batch, actual_y, actual_idxs = mdwm.get_batch_from_idx(4)
        exp_5_batch = np.array([
            [[2, 300, 310, 320, 3.00000, 0],
             [2, 301, 311, 321, 3.00000, 1],
             [2, 302, 312, 322, 3.00000, 2]]
        ])

        np.testing.assert_array_equal(x=exp_5_batch, y=actual_5_batch.detach().numpy(),
                                      err_msg='getting an actual batch didnt yield the expected values')

    def test_original_index_from_get_batch(self):
        mdwm = MovementDataWindowMaker(
            data=self.bigger_df,
            seq_identifier_col='id',
            labels_col='intId',
            window_size=3,
            data_is_sorted=True,
            sorting_cols=['id'],
            batch_size=1
        )

        actual_batch, actual_y, actual_idx = mdwm.get_batch_from_idx(1)
        expacted_samples = np.array([[[0, 101, 111, 121, 1.00000],
                                      [0, 102, 112, 122, 1.00000],
                                      [0, 103, 113, 123, 1.00000],
                                      ]])
        expacted_idx = [[1, 2, 3]]

        np.testing.assert_array_equal(np.array(expacted_idx), actual_idx, err_msg='expected different index labels')
        np.testing.assert_array_equal(expacted_samples, actual_batch, err_msg='the batch-data was not as expected')
        np.testing.assert_array_equal(np.array([1]), actual_y, err_msg='the y vales weren\'t as expected')

        data_from_idx = self.bigger_df.loc[actual_idx[0]]
        np.testing.assert_array_equal(expacted_samples,data_from_idx.values.reshape(1,3,-1),err_msg='using indexs returned form get_batch '
                                                                                    'as loc parameter dosnt yield the '
                                                                                    'expected data')

    def test_original_index_from_get_batch_with_sorting(self):
        shuffled_df = self.bigger_df
        shuffled_df['time'] = list(range(4)) * 3 + list(range(6)) + list(range(5))
        shuffled_df = shuffled_df.sample(frac=1,random_state=123456).reset_index(drop=True)
        mdwm = MovementDataWindowMaker(
            data=shuffled_df,
            seq_identifier_col='id',
            labels_col='intId',
            window_size=3,
            data_is_sorted=False,
            sorting_cols=['id','time'],
            batch_size=1
        )

        actual_batch, actual_y, actual_idx = mdwm.get_batch_from_idx(1)
        expacted_samples = np.array([[[0, 101, 111, 121, 1.00000,1],
                                      [0, 102, 112, 122, 1.00000,2],
                                      [0, 103, 113, 123, 1.00000,3],
                                      ]])
        expacted_idx = [[22, 8, 1]]

        np.testing.assert_array_equal(np.array(expacted_idx), actual_idx, err_msg='expected different index labels')
        np.testing.assert_array_equal(expacted_samples, actual_batch, err_msg='the batch-data was not as expected')
        np.testing.assert_array_equal(np.array([1]), actual_y, err_msg='the y vales weren\'t as expected')

        data_from_idx = shuffled_df.loc[actual_idx[0]]
        np.testing.assert_array_equal(expacted_samples, data_from_idx.values.reshape(1, 3, -1),
                                      err_msg='using indexs returned form get_batch '
                                              'as loc parameter dosnt yield the '
                                              'expected data on the origionally sorted df')
        data_from_idx = shuffled_df.loc[actual_idx[0]]
        np.testing.assert_array_equal(expacted_samples, data_from_idx.values.reshape(1, 3, -1),
                                      err_msg='using indexs returned form get_batch '
                                              'as loc parameter dosnt yield the '
                                              'expected data on the non-sorted df')
    # TODO big data tests..
    #  many frames (dtypes ok?)
    #  bigger windows / batches
    #  itterate a df multiple times?
    #  loc and iloc saftey tesets!!

    def test_shuffle_window_start_idxs(self):
        mdwm = MovementDataWindowMaker(
            data=self.bigger_df,
            seq_identifier_col='id',
            labels_col='intId',
            window_size=3,
            data_is_sorted=True,
            sorting_cols=['id'],
            batch_size=1,
            shuffle_windows=True
        )
        actual_batch, actual_y, actual_idx = mdwm.get_batch_from_idx(1)
        actual_batch = actual_batch.detach().numpy()
        # assert if samples within the window are sequential to each other
        self.assertTrue(actual_batch[0,0,1] <= actual_batch[0,1,1])
        self.assertTrue(actual_batch[0,1,1] <= actual_batch[0,2,1])


    def test_all_the_trouble(self):
        shuffled_df = self.bigger_df
        shuffled_df['time'] = list(range(4)) * 3 + list(range(6)) + list(range(5))
        shuffled_df = shuffled_df.sample(frac=1).reset_index(drop=True)
        mdwm = MovementDataWindowMaker(
            data=shuffled_df,
            seq_identifier_col='id',
            labels_col='intId',
            window_size=3,
            data_is_sorted=False,
            sorting_cols=['id', 'time'],
            batch_size=3,
            shuffle_windows=True
        )
        actual_batch, actual_y, actual_idx = mdwm.get_batch_from_idx(1)
        for i in range(2):
            # sequentilality of samples with a window
            self.assertTrue(actual_batch[i, 0, 1] <= actual_batch[i, 1, 1],f'\n{actual_batch[i]},\n{actual_batch[i+1]}')
            self.assertTrue(actual_batch[i, 1, 1] <= actual_batch[i, 2, 1],f'\n{actual_batch[i]},\n{actual_batch[i+1]}')
            # the actuall label for this window matches the label youd get pulling the sample from the original df with the idx form get_batch
            self.assertTrue(actual_y[i]==shuffled_df.loc[actual_idx[i],'intId'].iloc[0],f'{actual_y[i]} - {shuffled_df.loc[actual_idx[i],"intId"].iloc[0]}')


    def WindowMakerState(self, ex_last_window_idxs, ex_num_entries, ex_seq_lens, ex_seq_start_idxs,
                         ex_total_windows, ex_window_size, ex_window_start_idxs, ex_total_batches, mdwm):
        self.assertEqual(mdwm.num_entries, ex_num_entries, 'num_entries not set correct')
        self.assertEqual(mdwm.window_size, ex_window_size, 'window size not set correct')
        self.assertEqual(mdwm.total_num_batches, ex_total_batches, 'total number of batches available mismatch')
        # self.assertEqual(mdwm.sequenz_start_idxs,ex_seq_start_idxs,'seq_start_idxs not correct')
        np.testing.assert_array_equal(mdwm.sequenz_start_idxs, ex_seq_start_idxs)
        np.testing.assert_array_equal(mdwm.sequenz_lens, ex_seq_lens)
        np.testing.assert_array_equal(mdwm.last_window_start_idxs_per_sequenz, ex_last_window_idxs)
        np.testing.assert_array_equal(mdwm.window_start_idxs, ex_window_start_idxs)
        np.testing.assert_array_equal(mdwm.total_num_windows, ex_total_windows)


if __name__ == '__main__':
    unittest.main()
