from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ADE20K59ObjectDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 59 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    ## please note thta the class name is not right 
    CLASSES = (
        'building', 'sky', 'tree', 'bed ','windowpane', 'cabinet', 'person',
        'door', 'table', 'plant', 'chair', 'car', 'sofa', 'shelf', 'house', 'armchair', 'seat', 'fence', 'desk', 'wardrobe',
        'lamp', 'bathtub', 'column','chest of drawers', 'sink','skyscraper', 'refrigerator', 'boat',
        'stairs', 'pool table', 'stairway', 'bookcase', 'blind', 'coffee table',
        'toilet', 'stove','computer', 'swivel chair', 'bus', 'light', 'truck', 
        'chandelier','airplane', 'ottoman', 'bottle', 'van', 'washer', 'stool', 
         'minibike', 'oven', 'microwave', 'dishwasher', 'screen', 'hood', 'sconce',
        'traffic light', 'fan', 'glass', 'clock')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255]]

    def __init__(self, **kwargs):
        super(ADE20K59ObjectDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
