from typing import Optional, Any

import numpy as np
import pandas as pd
import torchmetrics.classification
from torch import Tensor
from sklearn.metrics import confusion_matrix

from src.datamodules.datasets.window_makers.sliding_window_maker import MovementDataWindowMaker


class BasicSequenceConfusionMatrix(torchmetrics.classification.ConfusionMatrix):
    def __init__(self,
                 num_classes: int,
                 window_maker: MovementDataWindowMaker,
                 normalize: Optional[str] = None,
                 threshold: float = 0.5,
                 multilabel: bool = False,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 file_name='basic_sequence_confusion_matrix.csv',
                 ):
        """

        """
        super().__init__(num_classes=num_classes, normalize=normalize, threshold=threshold, multilabel=multilabel,
                         compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)
        self.file_name = file_name
        self.window_maker = window_maker
        self.seq_id_col = window_maker.sequenz_identifier
        self.seq_id_list = window_maker.seq_id_list
        first_frame_idx = self.window_maker.window_size - 1
        self.seq_id_preds_targtes = pd.DataFrame(
            columns=['seq_id', 'predicted', 'target'], index=range(first_frame_idx, self.window_maker.num_entries))

    def add_batch(self, preds_batch, target_batch, sample_idxs):
        # preds_batch (batch_size, n_classes) <- softmax output
        # target_batch (batch_size)
        # sample_idxs (batch_size, windoiw_size) <- for each batch, windowsize[-1] would be the Frame, tht got predicted
        #
        # the idx for each predicted frame from this batch. Should be used to get the right row from windowmakers data df
        predicted_frames_idxs = sample_idxs[:, -1]
        predicted_frames_idxs = predicted_frames_idxs.detach().numpy()
        predicted_labels = preds_batch.argmax(axis=1)
        predicted_labels = predicted_labels.detach().numpy()
        target_batch_labels = target_batch.detach().numpy()
        self.seq_id_preds_targtes.loc[predicted_frames_idxs] = np.array([
            self.seq_id_list[predicted_frames_idxs],  # the seq_id for this window
            predicted_labels,  # the prediction for this window
            target_batch_labels  # the target for this window
        ]).T

    def compute_and_save_csv(self):
        preds = []#np.zeros((self.num_classes))
        targets = []#np.zeros((self.num_classes))
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

        conf_mat = confusion_matrix(y_true=targets, y_pred=preds)
        conf_df = pd.DataFrame(conf_mat, index=np.arange(self.num_classes), columns=np.arange(self.num_classes))
        conf_df.to_csv(self.file_name, index=False)

        # preds_majority_vote = self.seq_id_preds_targtes.groupby(self.seq_id_col).predicted.agg(pd.Series.mode)
        # pred_counts = preds_majority_vote.value_counts()
        # seq_preds[pred_counts.index] = pred_counts

        # targets_majority_vote = self.seq_id_preds_targtes.groupby(self.seq_id_col).target.agg(pd.Series.mode)
        # targets_counts = targets_majority_vote.value_counts()
        # seq_targets[targets_counts.index] = targets_counts
