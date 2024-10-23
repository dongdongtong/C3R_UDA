import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import timeit

source_perc = 30
target_perc = 60
def compute_dtm_single_label(img_gt, out_shape, unique_cls):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (num_cls, H, W)
    output: the Distance Transform Map (DTM) 
    dtm(x) = 0; x in segmentation boundary or out of segmentation
             inf|x-y|; x in segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_dtm = np.zeros(out_shape)
    source_lines = np.zeros(out_shape)
    target_lines = np.zeros(out_shape)
    boundaries = np.zeros(out_shape)

    for idx, c in enumerate(unique_cls):
        posmask = img_gt[c]
        negmask = 1 - posmask
        posdis = distance(posmask)
        boundary = skimage_seg.find_boundaries(posmask, mode='thick').astype(np.uint8)
        posdis[boundary==1] = 0

        source_split_intensity = np.percentile(posdis[img_gt[c] == 1], source_perc)
        boundary_source = skimage_seg.find_boundaries((posdis >= source_split_intensity), mode='inner').astype(np.uint8)
        source_lines[idx] = boundary_source

        target_split_intensity = np.percentile(posdis[img_gt[c] == 1], target_perc)
        boundary_target = skimage_seg.find_boundaries((posdis >= target_split_intensity), mode='thick').astype(np.uint8)
        target_lines[idx] = boundary_target

        gt_dtm[idx] = posdis
        boundaries[idx] = boundary
 
    return gt_dtm, boundaries, source_lines, target_lines


# visualization validation set: mr 24 index, npz 2285; ct 20 index, npz 727
data_npz_path = "data/validation_ct/62.npz"
data_npz = np.load(data_npz_path)
data = data_npz['slice']
label = data_npz['label']
print("label shape", label.shape)
print("label sum", label.sum(axis=(0, 1)))
print("unique label", np.unique(np.argmax(label, axis=-1)))
unique_label = np.unique(np.argmax(label, axis=-1))

colors = np.array([
    [0,     0,   0],
    [254, 232,  81], #LV-myo
    [145, 193,  62], #LA-blood
    [ 29, 162, 220], #LV-blood
    [238,  37,  36], #AA
    [  0,   0,   0], #Boundary Line
    [239, 141,  75], #Source Line
    [ 84, 130,  53], #Target Line
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

def overlay_img_dtm_line(img, seg):
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


gt_dtm_npy, boundaries, source_lines, target_lines = compute_dtm_single_label(label.transpose(2, 0, 1), (len(unique_label), 256, 256), unique_label)
# print("signed distance function time consume:", end - start)
print("max distance value", gt_dtm_npy.max())
print("min distance value", gt_dtm_npy.min())
# gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda(outputs_soft.device.index)

import matplotlib.pyplot as plt
from PIL import Image
import os

output_dir = "runs/analyse_distance_map"

label_argmax = np.argmax(label, axis=-1)
label_vis = decode_segmap(label_argmax)
print(label.shape, label_vis.shape)
# plot segmentation label
Image.fromarray(label_vis.astype(np.uint8)).save(os.path.join(output_dir, 'ct_label.png'))

slice_vis = np.squeeze(255*(data - data.min())/(data.max() - data.min() + 1e-6))
slice_seg_overlay = overlay_seg_img(slice_vis, label_argmax)
# plot slice segmentation overlay
Image.fromarray(slice_seg_overlay.astype(np.uint8)).save(os.path.join(output_dir, 'ct_slice_seg_overlay.png'))

# plot slice segmentation overlay with boundary line
label_argmax_boundary = label_argmax.copy()
for bd in boundaries:
    label_argmax_boundary[bd == 1] = 5
slice_seg_overlay = overlay_seg_img(slice_vis, label_argmax_boundary)
Image.fromarray(slice_seg_overlay.astype(np.uint8)).save(os.path.join(output_dir, 'ct_slice_seg_with_boundary_overlay.png'))

# get each class overlay png
for idx, c in enumerate(unique_label):
    label_c = label_argmax_boundary.copy()
    label_c[label_c != c] = 0
    label_c[boundaries[idx] == 1] = 5

    slice_labelc_overlay = overlay_seg_img(slice_vis, label_c)
    Image.fromarray(slice_labelc_overlay.astype(np.uint8)).save(os.path.join(output_dir, 'ct_slice_class_{}_overlay.png'.format(c)))


# plot slice segmentation overlay with source&target lines
label_argmax_dtm_lines = label_argmax_boundary.copy()
for source_line in source_lines:
    label_argmax_dtm_lines[source_line == 1] = 6
for target_line in target_lines:
    label_argmax_dtm_lines[target_line == 1] = 7

slice_seg_overlay = overlay_seg_img(slice_vis, label_argmax_dtm_lines)
Image.fromarray(slice_seg_overlay.astype(np.uint8)).save(os.path.join(output_dir, 'ct_slice_seg_with_boundary_dtm_lines_overlay.png'))

# get each class overlay png
for idx, c in enumerate(unique_label):
    label_c = label_argmax_dtm_lines.copy()
    label_c[label_c != c] = 0
    label_c[boundaries[idx] == 1] = 5
    label_c[source_lines[idx] == 1] = 6
    label_c[target_lines[idx] == 1] = 7

    slice_labelc_overlay = overlay_seg_img(slice_vis, label_c)
    Image.fromarray(slice_labelc_overlay.astype(np.uint8)).save(os.path.join(output_dir, 'ct_slice_class_dtm_lines_{}_overlay.png'.format(c)))

# plot a threshold line into the overlay png

# for idx, c in enumerate(unique_label):
#     if c == 0: # skip background
#         continue

#     dtm_c = gt_dtm_npy[idx]

#     source_threshold_intensity = np.percentile(dtm_c[label_argmax == c], 10)
#     target_threshold_intensity = np.percentile(dtm_c[label_argmax == c], 30)


#     label_c = label_argmax.copy()
#     label_c[label_c != c] = 0



