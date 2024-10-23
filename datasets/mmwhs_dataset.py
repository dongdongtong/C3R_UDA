import numpy as np

import torch
from torch.utils.data import Dataset


class MMHS(Dataset):
    def __init__(self, images_list, modality, transforms=None):
        super(Dataset, self).__init__()

        self.images_list = images_list
        self.transforms = transforms
        self.modality = modality

        if modality == 'ct':
            self.param1 = -2.8
            self.param2 = 3.2
        elif modality == 'mr':
            self.param1 = -1.8
            self.param2 = 4.4

    def __getitem__(self, index):
        # print("slice index: ", index)
        image_float_tensor, label_float_tensor = self._load_single_npz(self.images_list[index])

        if self.transforms:
            image_float_tensor = self.transforms(image_float_tensor)

        return image_float_tensor, label_float_tensor, index
    
    def _load_single_npz(self, img_pth):
        image_label_npz = np.load(img_pth)
        image = image_label_npz["slice"]
        label = image_label_npz["label"]

        image = 2.0*(image - self.param1)/(self.param2 - self.param1) - 1.0

        return torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)

    def __len__(self):
        return len(self.images_list)