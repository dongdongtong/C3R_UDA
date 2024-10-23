import os
from utils import utils
import time
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from io import BytesIO
import numpy as np
from torchvision.utils import make_grid

from PIL import Image

class Visualizer(object):
    def __init__(self, config):
        self.config = config
        
        self.log_dir = os.path.join(config['out_dir'], config['exp'], 'log')
        self.GEN_IMG_DIR = os.path.join(config['out_dir'], config['exp'], 'generated_imgs')
        self.writer = SummaryWriter(self.log_dir)

        self.valid_colors = [
            [  0,   0,  0],
            [254, 232, 81], # yellow LV-myo
            [145, 193, 62], # green LA-blood
            [ 29, 162, 220], # blue LV-blood
            [238,  37,  36]]  # Red AA
            
        self.label_colours = dict(zip(range(5), self.valid_colors))
    
    def decode_segmap(self, img):  # img is numpy.array object
        map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
        for idx in range(img.shape[0]):
            temp = img[idx, :, :]
            r = temp.copy()
            g = temp.copy()
            b = temp.copy()
            for l in range(0, 5):
                r[temp == l] = self.label_colours[l][0]
                g[temp == l] = self.label_colours[l][1]
                b[temp == l] = self.label_colours[l][2]

            rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0
            map[idx, :, :, :] = rgb
        return map

    def display_current_results(self, visuals, domain, step):
        imgs = []
        for key, t in visuals.items():
            # resize the visulas to 256x256 image
            # t_resize = F.interpolate(t, (256, 256), mode='bicubic')
            if 'seg' in key:
                img_seg = self.decode_segmap(t)
                imgs.append(torch.tensor(img_seg.transpose((0, 3, 1, 2)), dtype=torch.float))
            else:
                imgs.append((t.expand(-1, 3, -1, -1).cpu() + 1) / 2)
        
        imgs = torch.cat(imgs, 0)     #Concatenates the given sequence of seq tensors in the given dimension.
        imgs = make_grid(imgs.detach(), nrow=self.config['batch_size'], normalize=False, scale_each=True).cpu().numpy()   #Make a grid of images.
        imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)   #限制数组值在一定范围 若小于0 则变为0
        imgs = imgs.transpose((1, 2, 0))
        imgs = Image.fromarray(imgs)
        filename = '%05d_%s.jpg' % (step, "source" if domain=="s" else "target")
        imgs.save(os.path.join(self.GEN_IMG_DIR, filename))

    def plot_current_errors(self, errors, step):
        
        for tag, value in errors.items():
            if tag == "cur_evaluate_dice":
                pass
            else:
                value = value.mean().float()
            self.writer.add_scalar(tag, value, step)

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %s) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
