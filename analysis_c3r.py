import os

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image


colors = np.array([
    [0,     0,   0],
    [254, 232,  81], #LV-myo
    [145, 193,  62], #LA-blood
    [ 29, 162, 220], #LV-blood
    [238,  37,  36], #AA
    ])

def decode_segmap(seg_pred):  # seg_pred is numpy.array object, shape of (H, W)
    temp = seg_pred.copy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 5):
        r[temp == l] = colors[l][0]
        g[temp == l] = colors[l][1]
        b[temp == l] = colors[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

def overlay_seg_img(img, seg):
    # get unique labels
    labels = np.unique(seg)

    # remove background
    labels = labels[labels !=0]

    # img backgournd
    img_b = img*(seg == 0)

    # final_image
    final_img = np.zeros([img.shape[0], img.shape[1], 3])

    final_img += img_b[:, :, np.newaxis]

    for l in labels:
        mask = seg == l
        img_f = img*mask

        # area of labeled l img, convert to rgb
        img_f = np.tile(img_f, (3, 1, 1)).transpose(1, 2, 0)

        # colored segmentation
        img_seg = colors[l*mask]

        # alpha overlay
        final_img += 0.5*img_f + 0.5*img_seg
    
    return final_img

def get_overlaped_outputs(outputs, labels):
    outputs_copy = outputs.copy()
    outputs[outputs_copy >= 0.5] = 1
    outputs[outputs_copy < 0.5] = 0
    # print("outputs shape:", outputs.shape, "labels shape:", labels.shape, "outputs sum:", outputs.sum(), "labels sum:", labels.sum())

    overlap_out_gt = np.zeros(outputs.shape).astype(np.int)

    true_positive = outputs * labels
    false_negative = labels - true_positive
    false_positive = outputs - true_positive
    overlap_out_gt[true_positive.astype(np.int) == 1] = 1  # overlap area labeled 1  green
    overlap_out_gt[false_negative.astype(np.int) == 1] = 2  # false negative area labeled 2  gold   
    overlap_out_gt[false_positive.astype(np.int) == 1] = 3  # false positive labeled 3   red

    return overlap_out_gt

# configure devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
cuda = torch.cuda.is_available()


from models.seg_model import DilateResUNetCLMem, ResUNetDecoder

# --------------------
# Construct Networks
# --------------------
pretrain_model_path = "runs/complete_ablation/runs/config_mr2ct/ITDFN+D_p+CCFA+CR/55389/models/23999"
seg_net = DilateResUNetCLMem(n_channels=1, n_classes=5, norm="InstanceNorm", act="leaky_relu", latent_dims=128)
seg_net_ema = DilateResUNetCLMem(n_channels=1, n_classes=5, norm="InstanceNorm", act="leaky_relu", latent_dims=128)
decoder_T = ResUNetDecoder(1)
decoder_S = ResUNetDecoder(1)
seg_net.load_state_dict(torch.load(os.path.join(pretrain_model_path, "base_model_23999.pt")))
seg_net_ema.load_state_dict(torch.load(os.path.join(pretrain_model_path, "base_model_ema_23999.pt")))
decoder_T.load_state_dict(torch.load(os.path.join(pretrain_model_path, "decoder_T_23999.pt")))
decoder_S.load_state_dict(torch.load(os.path.join(pretrain_model_path, "decoder_S_23999.pt")))



if cuda:
    seg_net = seg_net.cuda()
    seg_net_ema = seg_net_ema.cuda()
    decoder_T = decoder_T.cuda()
    decoder_S = decoder_S.cuda()
    #
    # encoder = encoder.cuda()
    # segmenter = segmenter.cuda()

    # model = model.cuda()

if torch.cuda.device_count() > 1:
    pass
    # ganunetv4 = nn.DataParallel(ganunetv4, device_ids=device_ids)
    # encoder = nn.DataParallel(encoder, device_ids=device_ids)
    # segmenter = nn.DataParallel(segmenter, device_ids=device_ids)
    # model = nn.DataParallel(model, device_ids=device_ids)


from glob import glob
test_list = list(glob(os.path.join("data/test_ct", "*.npz")))


# --------------------
# Evaluate model
# --------------------

# --------------------
# Train model
# --------------------
seg_net.eval()
seg_net_ema.eval()
decoder_T.eval()
decoder_S.eval()


range_src = (-1.8, 4.4)  # MR
range_tar = (-2.8, 3.2)  # CT

output_mask_dir = "runs/analyse_segmentation"
os.makedirs(output_mask_dir, exist_ok=True)


from datasets import create_dataset
import yaml
from tensorboardX import SummaryWriter
from utils.visualizer import Visualizer
from utils.utils import get_logger
with open("configs/config_mr2ct_ITDFN_cl_mem_dtm_analysis_seg.yml") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

# path = cfg['training']['save_path']
writer = SummaryWriter(log_dir=output_mask_dir)
visual = Visualizer(cfg, output_mask_dir, writer)

print('RUNDIR: {}'.format(output_mask_dir))

logger = get_logger(output_mask_dir)
logger.info('Begin visulizing segmentations on validation set')
## create dataset
datasets = create_dataset(cfg, writer, logger)

mr_valid_loader = datasets.source_valid_loader
ct_valid_loader = datasets.target_valid_loader

print(datasets.source_valid.files[184])
print(datasets.target_valid.files[184])

i = 0
with torch.no_grad():
    zip_source_target_train_loader = zip(mr_valid_loader, ct_valid_loader)

    for source_batch, target_batch in zip_source_target_train_loader:

        if i == 245:
            break

        source_images, source_labels, source_indexes = source_batch
        target_images, target_labels, target_indexes = target_batch

        print(int(source_indexes), int(target_indexes), datasets.source_valid.files[source_indexes], datasets.target_valid.files[source_indexes])

        source_images = source_images.cuda()
        target_images = target_images.cuda()

        source_out_dict = seg_net(source_images)
        target_out_dict = seg_net(target_images)
        target_ema_out_dict = seg_net_ema(target_images)

        fake_T = decoder_T(source_out_dict["multi_lv_feats"])
        fake_S = decoder_S(target_out_dict["multi_lv_feats"])

        fake_T_out_dict = seg_net(fake_T)
        fake_S_out_dict = seg_net(fake_S)

        cyc_S = decoder_S(fake_T_out_dict["multi_lv_feats"])
        cyc_T = decoder_T(fake_S_out_dict["multi_lv_feats"])

        source_pred = torch.argmax(torch.softmax(source_out_dict["seg"], dim=1), dim=1).squeeze().data.cpu().numpy()
        target_pred = torch.argmax(torch.softmax(target_out_dict["seg"], dim=1), dim=1).squeeze().data.cpu().numpy()
        target_ema_pred = torch.argmax(torch.softmax(target_ema_out_dict["seg"], dim=1), dim=1).squeeze().data.cpu().numpy()
        source_pred_vis = decode_segmap(source_pred)
        target_pred_vis = decode_segmap(target_pred)
        target_pred_ema_vis = decode_segmap(target_ema_pred)

        fake_T_pred = torch.argmax(torch.softmax(fake_T_out_dict["seg"], dim=1), dim=1).squeeze().data.cpu().numpy()
        fake_S_pred = torch.argmax(torch.softmax(fake_S_out_dict["seg"], dim=1), dim=1).squeeze().data.cpu().numpy()
        fake_T_pred_vis = decode_segmap(fake_T_pred)
        fake_S_pred_vis = decode_segmap(fake_S_pred)

        source_slice = source_images.squeeze().data.cpu().numpy()
        target_slice = target_images.squeeze().data.cpu().numpy()
        source_slice_vis = 255*(source_slice - source_slice.min())/(source_slice.max() - source_slice.min() + 1e-6)
        target_slice_vis = 255*(target_slice - target_slice.min())/(target_slice.max() - target_slice.min() + 1e-6)


        source_slice_seg_overlay = overlay_seg_img(source_slice_vis, source_pred.astype(np.int))
        target_slice_seg_overlay = overlay_seg_img(target_slice_vis, target_pred.astype(np.int))
        target_slice_emaseg_overlay = overlay_seg_img(target_slice_vis, target_ema_pred.astype(np.int))

        # for labels overlay
        source_labels_numpy = torch.argmax(source_labels, dim=-1).squeeze().numpy()
        target_labels_numpy = torch.argmax(target_labels, dim=-1).squeeze().numpy()
        source_label_vis = decode_segmap(source_labels_numpy)
        target_label_vis = decode_segmap(target_labels_numpy)
        source_slice_label_overlay = overlay_seg_img(source_slice_vis, source_labels_numpy.astype(np.int))
        target_slice_label_overlay = overlay_seg_img(target_slice_vis, target_labels_numpy.astype(np.int))

        fake_T_slice = fake_T.squeeze().data.cpu().numpy()
        fake_S_slice = fake_S.squeeze().data.cpu().numpy()
        cyc_T_slice = cyc_T.squeeze().data.cpu().numpy()
        cyc_S_slice = cyc_S.squeeze().data.cpu().numpy()
        fake_T_slice_vis = 255*(fake_T_slice - fake_T_slice.min())/(fake_T_slice.max() - fake_T_slice.min() + 1e-6)
        fake_S_slice_vis = 255*(fake_S_slice - fake_S_slice.min())/(fake_S_slice.max() - fake_S_slice.min() + 1e-6)
        cyc_S_slice_vis = 255*(cyc_S_slice - cyc_S_slice.min())/(cyc_S_slice.max() - cyc_S_slice.min() + 1e-6)
        cyc_T_slice_vis = 255*(cyc_T_slice - cyc_T_slice.min())/(cyc_T_slice.max() - cyc_T_slice.min() + 1e-6)

        fake_T_slice_seg_overlay = overlay_seg_img(fake_T_slice_vis, fake_T_pred.astype(np.int))
        fake_S_slice_seg_overlay = overlay_seg_img(fake_S_slice_vis, fake_S_pred.astype(np.int))

        source_slice_vis = np.tile(source_slice_vis, (3, 1, 1)).transpose(1, 2, 0)
        target_slice_vis = np.tile(target_slice_vis, (3, 1, 1)).transpose(1, 2, 0)
        fake_T_slice_vis = np.tile(fake_T_slice_vis, (3, 1, 1)).transpose(1, 2, 0)
        fake_S_slice_vis = np.tile(fake_S_slice_vis, (3, 1, 1)).transpose(1, 2, 0)
        cyc_S_slice_vis = np.tile(cyc_S_slice_vis, (3, 1, 1)).transpose(1, 2, 0)
        cyc_T_slice_vis = np.tile(cyc_T_slice_vis, (3, 1, 1)).transpose(1, 2, 0)

        dir_name = os.path.join(output_mask_dir, str(i))
        os.makedirs(dir_name, exist_ok=True)
        # original/translated/cycle images
        Image.fromarray(source_slice_vis.astype(np.uint8)).save(os.path.join(dir_name, 'mr_orig_image_{}.png'.format(int(source_indexes))))
        Image.fromarray(target_slice_vis.astype(np.uint8)).save(os.path.join(dir_name, 'ct_orig_image_{}.png'.format(int(target_indexes))))
        Image.fromarray(fake_T_slice_vis.astype(np.uint8)).save(os.path.join(dir_name, 'mr_trans_image_{}.png'.format(int(source_indexes))))
        Image.fromarray(fake_S_slice_vis.astype(np.uint8)).save(os.path.join(dir_name, 'ct_trans_image_{}.png'.format(int(source_indexes))))
        Image.fromarray(cyc_S_slice_vis.astype(np.uint8)).save(os.path.join(dir_name, 'mr_cyc_image_{}.png'.format(int(source_indexes))))
        Image.fromarray(cyc_T_slice_vis.astype(np.uint8)).save(os.path.join(dir_name, 'ct_cyc_image_{}.png'.format(int(source_indexes))))
        
        # only pred and label visualize
        Image.fromarray(source_pred_vis.astype(np.uint8)).save(os.path.join(dir_name, 'mr_seg_{}.png'.format(int(source_indexes))))
        Image.fromarray(target_pred_vis.astype(np.uint8)).save(os.path.join(dir_name, 'ct_seg_{}.png'.format(int(source_indexes))))
        Image.fromarray(target_pred_ema_vis.astype(np.uint8)).save(os.path.join(dir_name, 'ct_emaseg_{}.png'.format(int(source_indexes))))
        Image.fromarray(fake_T_pred_vis.astype(np.uint8)).save(os.path.join(dir_name, 'mr_trans_seg_{}.png'.format(int(source_indexes))))
        Image.fromarray(fake_S_pred_vis.astype(np.uint8)).save(os.path.join(dir_name, 'ct_trans_seg_{}.png'.format(int(source_indexes))))
        Image.fromarray(source_label_vis.astype(np.uint8)).save(os.path.join(dir_name, 'mr_label_{}.png'.format(int(source_indexes))))
        Image.fromarray(target_label_vis.astype(np.uint8)).save(os.path.join(dir_name, 'ct_label_{}.png'.format(int(source_indexes))))
        
        # overlay images
        Image.fromarray(source_slice_seg_overlay.astype(np.uint8)).save(os.path.join(dir_name, 'mr_orig_image_seg_overlay_{}.png'.format(int(source_indexes))))
        Image.fromarray(target_slice_seg_overlay.astype(np.uint8)).save(os.path.join(dir_name, 'ct_orig_image_seg_overlay_{}.png'.format(int(source_indexes))))
        Image.fromarray(target_slice_emaseg_overlay.astype(np.uint8)).save(os.path.join(dir_name, 'ct_orig_image_emaseg_overlay_{}.png'.format(int(source_indexes))))
        Image.fromarray(fake_T_slice_seg_overlay.astype(np.uint8)).save(os.path.join(dir_name, 'mr_trans_image_seg_overlay_{}.png'.format(int(source_indexes))))
        Image.fromarray(fake_S_slice_seg_overlay.astype(np.uint8)).save(os.path.join(dir_name, 'ct_trans_image_seg_overlay_{}.png'.format(int(source_indexes))))
        Image.fromarray(source_slice_label_overlay.astype(np.uint8)).save(os.path.join(dir_name, 'mr_orig_image_label_overlay_{}.png'.format(int(source_indexes))))
        Image.fromarray(target_slice_label_overlay.astype(np.uint8)).save(os.path.join(dir_name, 'ct_orig_image_label_overlay_{}.png'.format(int(source_indexes))))
        
        

        i += 1