from mmseg.datasets import CityscapesDataset, CustomDataset

from .builder import DATASETS


@DATASETS.register_module()
class CidaDataset(CustomDataset):

    
    CLASSES = ('Road', 'Sidewalk', 'Construction', 'Fence', 'Pole',
                'Traffic Light', 'Traffic Sign', 'Nature', 'Sky', 'Person',
                'Rider', 'Car'),
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [190, 153, 153], [153, 153, 153],
                [250, 170, 30], [220, 220, 0], [107, 142, 35], [70, 130, 180], [220, 20, 60],
                [255, 0, 0], [0, 0, 142]]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)