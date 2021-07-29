from typing import Optional, Any

import numpy as np
import pandas as pd
import torchmetrics.classification
import seaborn as sn
from matplotlib import pyplot as plt
from torch import Tensor
from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_compute


class MyConfusionMatrix(torchmetrics.classification.ConfusionMatrix):
    def __init__(self,
                 num_classes: int,
                 normalize: Optional[str] = None,
                 threshold: float = 0.5,
                 multilabel: bool = False,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 file_name='confusion_matrix.csv',
                 ):
        super().__init__(num_classes=num_classes, normalize=normalize, threshold=threshold, multilabel=multilabel,
                         compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)
        self.file_name = file_name

    def compute_and_save_csv(self) -> Tensor:
        result = _confusion_matrix_compute(self.confmat, self.normalize)
        conf_mat = result.detach().cpu().numpy().astype(np.int)
        conf_df = pd.DataFrame(conf_mat, index=np.arange(self.num_classes), columns=np.arange(self.num_classes))
        conf_df.to_csv(self.file_name, index=False)
        return result
