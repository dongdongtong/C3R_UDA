"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
# from torch.nn.modules.module import T
from .augmentations import get_composed_augmentations
import torch.utils.data
import numpy as np
# import data.cityscapes_dataset

def find_dataset_using_name(name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = name + '_loader'
    for _name, cls in datasetlib.__dict__.items():
        if _name.lower() == target_dataset_name.lower():
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(cfg, writer, logger):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(cfg, writer, logger)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, cfg, writer, logger):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        # self.opt = opt
        self.cfg = cfg
        # self.name = name
        # self.status = status
        self.writer = writer
        self.logger = logger

        # status == 'train':
        cfg_source = cfg['data']['source']
        # print("cfg source", cfg_source)
        cfg_target = cfg['data']['target']
        # print("cfg target", cfg_target)

        source_train = find_dataset_using_name(cfg_source['name'])
        augmentations = cfg_source.get('transforms', None)
        transforms = get_composed_augmentations(augmentations)
        self.source_train = source_train(cfg_source, transforms=transforms)
        logger.info("{} source dataset has been created".format(self.source_train.__class__.__name__))
        print("dataset {} for source was created".format(self.source_train.__class__.__name__))

        target_train = find_dataset_using_name(cfg_target['name'])
        augmentations = cfg_target.get('transforms', None)
        transforms = get_composed_augmentations(augmentations)
        self.target_train = target_train(cfg_target, transforms=transforms)
        logger.info("{} target dataset has been created".format(self.target_train.__class__.__name__))
        print("dataset {} for target was created".format(self.target_train.__class__.__name__))

        def worker_init_fn(worker_id):
            # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
            print("dataloader worker ", worker_id, " numpy random seed is set to ", 88 + worker_id)
            np.random.seed(88 + worker_id)

        ## create train loader
        self.source_train_loader = torch.utils.data.DataLoader(
            dataset=self.source_train,
            batch_size=cfg_source['batch_size'],
            shuffle=True,
            num_workers=int(cfg['data']['num_workers']),
            drop_last=True,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
        )
        self.target_train_loader = torch.utils.data.DataLoader(
            self.target_train,
            batch_size=cfg_target['batch_size'],
            shuffle=True,
            num_workers=int(cfg['data']['num_workers']),
            drop_last=True,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
        )

        # status == valid
        cfg_source_valid = cfg['data']['source_valid']
        self.source_valid = None
        self.source_valid_loader = None
        if cfg_source_valid != None:
            source_valid = find_dataset_using_name(cfg_source_valid['name'])
            self.source_valid = source_valid(cfg_source_valid, transforms=None)
            logger.info("{} source_valid dataset has been created".format(self.source_valid.__class__.__name__))
            print("dataset {} for source_valid was created".format(self.source_valid.__class__.__name__))

            self.source_valid_loader = torch.utils.data.DataLoader(
                self.source_valid,
                batch_size=cfg_source_valid['batch_size'],
                shuffle=cfg_source_valid['shuffle'],
                num_workers=int(cfg['data']['num_workers']),
                drop_last=False,
                pin_memory=False,
            )

        self.target_valid = None
        self.target_valid_loader = None
        cfg_target_valid = cfg['data']['target_valid']
        if cfg_target_valid != None:
            target_valid = find_dataset_using_name(cfg_target_valid['name'])
            self.target_valid = target_valid(cfg_target_valid, transforms=None)
            logger.info("{} target_valid dataset has been created".format(self.target_valid.__class__.__name__))
            print("dataset {} for target_valid was created".format(self.target_valid.__class__.__name__))

            self.target_valid_loader = torch.utils.data.DataLoader(
                self.target_valid,
                batch_size=cfg_target_valid['batch_size'],
                shuffle=cfg_target_valid['shuffle'],
                num_workers=int(cfg['data']['num_workers']),
                drop_last=False,
                pin_memory=False,
            )

        logger.info("train and valid dataset has been created")
        print("train and valid dataset has been created")

    def load_data(self):
        return self

