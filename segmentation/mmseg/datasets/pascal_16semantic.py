from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Pascal16SemanticDataset(CustomDataset):
    """LIP dataset.
    In segmentation map annotation for LIP, 0 stands for background, which
    is not included in 19 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    ## please note thta the class name is not right 
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'cat', 'cow', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor')

    PALETTE = [
        [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [143, 255, 140],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               ]

    def __init__(self, **kwargs):
        super(Pascal16SemanticDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)