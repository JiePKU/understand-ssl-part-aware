from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LIPDataset(CustomDataset):
    """LIP dataset.
    In segmentation map annotation for LIP, 0 stands for background, which
    is not included in 19 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    ## please note thta the class name is not right 
    CLASSES = (
        'Hat', 'Hair', 'Glove', 'Sunglasses', 'UpperClothes', 'Dress', 'Coat', 'Socks',
        'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
        'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7]]

    def __init__(self, **kwargs):
        super(LIPDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)