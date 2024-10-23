import numpy as np

import torch
import random
from torch.utils.data import Dataset

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class mr_loader(Dataset):
    def __init__(self, config, transforms=None):
        super(mr_loader, self).__init__()

        self.config = config
        self.root = config['rootpath']

        # read txt files from file name
        with open(config['rootpath'], 'r') as fp:
            samples_txt = fp.readlines()
        self.files = [row[:-1] for row in samples_txt]

        # img data param
        self.img_size = (
            config['size_H'], config['size_W']
        )

        # img data param
        self.param1 = config['param1']
        self.param2 = config['param2']

        # augmentation param
        self.transforms = transforms
        self.is_transform = config.get('is_transform', False)
        self.p = config.get('aug_p', 0.5)

    def __getitem__(self, index):
        # print("slice index: ", index)
        image_float_tensor, label_float_tensor = self._load_single_npz(self.files[index])

        return image_float_tensor, label_float_tensor, index
    
    def _load_single_npz(self, img_pth):
        image_label_npz = np.load(img_pth)
        image = image_label_npz["slice"]
        # label = image_label_npz["label"].argmax(axis=-1)
        label = image_label_npz["label"]

        image = 2.0*(image - self.param1)/(self.param2 - self.param1) - 1.0

        if self.transforms is not None and self.is_transform:
            if self.p > random.random():
                segmap = label[:, :, None].astype(np.int32)
                segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
                image, label = self.transforms(image=image, segmentation_maps=segmap)
                # print(type(image), image.shape, type(label), label.shape)
                label = label.get_arr().squeeze(axis=-1)

        # return torch.from_numpy(image).type(torch.FloatTensor).permute(2, 0, 1), torch.from_numpy(label).type(torch.LongTensor)
        return torch.from_numpy(image).type(torch.FloatTensor).permute(2, 0, 1), torch.from_numpy(label).type(torch.FloatTensor)

    def __len__(self):
        return len(self.files)