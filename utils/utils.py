import torch
import numpy as np

import logging
import os
import datetime

def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class LambdaLROnStep:
    def __init__(self, n_steps, offset_step, decay_start_step, min_lr_rate):
        assert (n_steps - decay_start_step) > 0, "Decay must start before the training session ends!"
        self.n_steps = n_steps
        self.offset_step = offset_step
        self.decay_start_step = decay_start_step
        self.min_lr_rate = min_lr_rate

    def step(self, step):
        return max(1.0 - max(0, step + self.offset_step - self.decay_start_step) / (self.n_steps - self.decay_start_step), self.min_lr_rate)

class PolyLR:
    def __init__(self, n_steps, offset_step, decay_start_step, power=0.9):
        assert (n_steps - decay_start_step) > 0, "Decay must start before the training session ends!"
        self.n_steps = n_steps
        self.offset_step = offset_step
        self.decay_start_step = decay_start_step
        self.power = power

    def step(self, step):
        return (1 - float(step) / self.n_steps) ** (self.power)


def get_scheduled_ema_consistency(cur_step, n_steps, lambda_consistency, offset=0):
    return lambda_consistency * ((cur_step + 1 - offset) / (n_steps - offset))


def dice_score(logits, gts, smooth=1.):
    logits = logits.permute((0, 2, 3, 1))
    dice = 0
    softmaxpred = torch.softmax(logits, dim=-1)

    for i in range(5):
        inse = torch.sum(softmaxpred[:, :, :, i] * gts[:, :, :, i]) + smooth
        l = torch.sum(softmaxpred[:, :, :, i])
        r = torch.sum(gts[:, :, :, i])
        dice += 2.0 * inse / (l + r + smooth)

    # print("DiceLoss shape:", dice)

    return dice / 5


def one_hot(targets):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), 5, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    
    return one_hot


def min_max_norm(tensor):
    out = tensor.clone()
    min = tensor.min()
    max = tensor.max()
    # print("min: {}, max: {}, max - min: {}".format(min, max, max-min))
    out = (out - min) / (max - min)
    return  out


def loop_iterable(iterable):
    while True:
        yield from iterable


def color_seg(seg):

    colors = torch.tensor([
    [0,     0,   0],
    [254, 232,  81], #LV-myo
    [145, 193,  62], #LA-blood
    [ 29, 162, 220], #LV-blood
    [238,  37,  36]]) #AA

    out = []
    for s in seg:
        out.append((colors[s[0]]).permute(2, 0, 1))
    return torch.stack(out, dim=0)


def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=True):
    def tile_images(imgs, picturesPerRow=4):
        if imgs.shape[0] % picturesPerRow == 0:
            rowPadding = 0
        else:
            rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
        if rowPadding > 0:
            imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

        # Tiling Loop (The conditionals are not necessary anymore)
        tiled = []
        for i in range(0, imgs.shape[0], picturesPerRow):
            tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

        tiled = np.concatenate(tiled, axis=0)
        return tiled

    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image, imtype, normalize)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np
        
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / (2) * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


@torch.no_grad()
def one_hot(targets, n_classes):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), n_classes, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    
    return one_hot


def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist
#返回单个dice值
def dice_eval(label_pred, label_true, n_class):
    hist = fast_hist(label_true, label_pred, n_class)   #;print(hist)
    union = hist.sum(axis=0) + hist.sum(axis=1)
    dice = (2 * np.diag(hist) + 1e-8) / (union + 1e-8)
    dice[union == 0] = np.nan
    return dice


def overlay_seg_anchor_sampling(seg, selected_indices):
    # seg: argmax label, shape: 2D (H, W), type: numpy.nd_array
    # selected_indices: 1D array, type: numpy.nd_array
    colors = np.array([
    [0,     0,   0],
    [254, 232,  81], #LV-myo
    [145, 193,  62], #LA-blood
    [ 29, 162, 220], #LV-blood
    [238,  37,  36], #AA
    [0,     0,   0],]) 

    seg_selected_pixels = np.zeros([seg.shape[0], seg.shape[1]]).flatten()
    seg_selected_pixels[selected_indices] = 255
    seg_selected_pixels = seg_selected_pixels.reshape((seg.shape[0], seg.shape[1]))

    # get unique labels
    labels = np.unique(seg)

    # remove background
    labels = labels[labels !=0]

    # backgournd anchors selected
    bg_sel_anchors = seg_selected_pixels*(seg == 0)

    # final_image
    final_img = np.zeros([seg.shape[0], seg.shape[1], 3])

    final_img += bg_sel_anchors[:, :, np.newaxis]

    for l in labels:
        mask = seg == l
        img_f = seg_selected_pixels*mask

        # convert to rgb
        img_f = np.tile(img_f, (3, 1, 1)).transpose(1, 2, 0)

        # colored segmentation
        img_seg = colors[(l*mask).astype(int)]

        # alpha overlay
        final_img += 0.5*img_f + 0.5*img_seg
    
    return final_img / 255.0