import logging

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

logger = logging.getLogger('ptsemseg')

def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    transforms = None
    for aug_key, aug_param in aug_dict.items():
        if aug_key == "zoom":
            transforms = iaa.Sequential([
                    iaa.Affine(scale=aug_param, seed=88)
                ])
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return transforms