from email.policy import default
import os
import sys
import yaml
import time
import shutil
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)

from torch.utils import data
from tqdm import tqdm
from datasets import create_dataset
from utils.utils import get_logger
import models.c3r_trainer as c3r_trainer
from utils.visualizer import Visualizer
from tensorboardX import SummaryWriter
import itertools
from evaluate import evaluate_ct, evaluate_mr, evaluate_ct_val, evaluate_mr_val

def train(cfg, writer, logger, visual, logdir):
    # fix random seed
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(cfg.get('seed', 88)) 
    np.random.seed(cfg.get('seed', 88))
    torch.manual_seed(cfg.get('seed', 88))
    torch.random.manual_seed(cfg.get('seed', 88))
    torch.cuda.manual_seed(cfg.get('seed', 88))
    torch.cuda.manual_seed_all(cfg.get('seed', 88))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    ## create dataset
    datasets = create_dataset(cfg, writer, logger)  #source_train\ target_train\ source_valid\ target_valid + _loader

    # trainer
    model = c3r_trainer.AdaptationModel(cfg, writer, logger, visual, logdir)

    random.seed(cfg.get('seed', 88)) 
    np.random.seed(cfg.get('seed', 88))
    torch.manual_seed(cfg.get('seed', 88))
    torch.random.manual_seed(cfg.get('seed', 88))
    torch.cuda.manual_seed(cfg.get('seed', 88))
    torch.cuda.manual_seed_all(cfg.get('seed', 88))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False   # we turn to false as it slow down the training and sometimes causes OOM

    # Setup Metrics
    flag_train = True

    epoches = cfg['training']['n_epoches']

    source_train_loader = datasets.source_train_loader
    target_train_loader = datasets.target_train_loader
    epoch_batches = max(len(source_train_loader), len(target_train_loader))
    model.init_lr_schedulers(epoch_batches)
    logger.info('source train batchsize is {}'.format(source_train_loader.batch_size))
    logger.info('source train loader len is {}'.format(len(source_train_loader)))
    print('source train batchsize is {}'.format(source_train_loader.batch_size))
    print('source train loader len is {}'.format(len(source_train_loader)))
    logger.info('target train batchsize is {}'.format(target_train_loader.batch_size))
    logger.info('target train loader len is {}'.format(len(target_train_loader)))
    print('target train batchsize is {}'.format(target_train_loader.batch_size))
    print('target train loader len is {}'.format(len(target_train_loader)))

    val_loader = None
    if cfg.get('valset') == 'mr':
        val_loader = datasets.source_valid_loader
        logger.info('valset is mr')
        print('valset is mr')
    else:
        val_loader = datasets.target_valid_loader
        logger.info('valset is ct')
        print('valset is ct')
    logger.info('val batchsize is {}'.format(val_loader.batch_size))
    print('val batchsize is {}'.format(val_loader.batch_size))

    # begin training
    model.iter = 0
    prev_time = time.time()
    for epoch in range(epoches):
        if not flag_train:
            break
        if model.iter > cfg['training']['train_iters']:
            break

        if len(source_train_loader) > len(target_train_loader):
            zip_source_target_train_loader = zip(source_train_loader, itertools.cycle(target_train_loader))
        elif len(source_train_loader) < len(target_train_loader):
            zip_source_target_train_loader = zip(itertools.cycle(source_train_loader), target_train_loader)
        else:
            zip_source_target_train_loader = zip(source_train_loader, target_train_loader)

        for source_batch, target_batch in zip_source_target_train_loader:
            i = model.iter
            if i > cfg['training']['train_iters']:
                break

            source_images, source_labels, source_indexes = source_batch
            target_images, target_labels, target_indexes = target_batch

            source_images = source_images.cuda()
            source_labels = source_labels.cuda()
            target_images = target_images.cuda()
            target_labels = target_labels.cuda()

            model.set_input(source_images, source_labels, target_images, target_labels)
            model.train()
            model.step()

            batches_left = epoches * epoch_batches - model.iter
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Seg loss: %4f] ETA: %s"
                % (
                    epoch+1,
                    epoches,
                    model.iter,
                    epoches * epoch_batches,
                    0.,
                    time_left,
                )
            )

            # evaluation
            if i % cfg['training']['val_interval'] == 0 or \
                (i + 1) == cfg['training']['train_iters']:
                model.eval()
                model.visualization()

                torch.cuda.empty_cache()

                with torch.no_grad():
                    eval_ct_dice = evaluate_ct(model.seg_net_DP, cfg)
                    eval_mr_dice = evaluate_mr(model.seg_net_DP, cfg)
                    writer.add_scalar('eval_dice/ct_dice', eval_ct_dice, model.iter)
                    writer.add_scalar('eval_dice/mr_dice', eval_mr_dice, model.iter)
                    logger.info('%05d eval_ct_dice: %.4f'% (model.iter, eval_ct_dice))
                    logger.info('%05d eval_mr_dice: %.4f'% (model.iter, eval_mr_dice))
                    
                    # you can insert evaluation code on the validation set, ...
                    
                    # save the last model
                    if (i + 1) == cfg['training']['train_iters']:
                        model_dir = os.path.join(logdir, "models", str(model.iter))
                        os.makedirs(model_dir, exist_ok=True)
                        model.save(model_dir)
                        
                torch.cuda.empty_cache()
            model.iter += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/config_mr2ct_ITDFN_cl_mem_dtm.yml',
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    run_id = random.randint(1, 100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)
    visual = Visualizer(cfg, logdir, writer)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    train(cfg, writer, logger, visual, logdir)
