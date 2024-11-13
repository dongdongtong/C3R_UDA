import os
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

import nibabel as nib
import cv2

np.random.seed(88)
torch.manual_seed(88)


colors = np.array([
    [0,    0,  0],  # background
    [ 81,  0, 81],  # spleen
    [128, 64,128],  # right kidney
    [244, 35,232],  # left kidney
    [250,170,160],  # gallbladder
    [230,150,140],  # esophagus
    [ 70, 70, 70],  # liver
    [102,102,156],  # stomach
    [190,153,153],  # aorta
    [180,165,180],  # inferior vena cava
    [150,100,100],  # pancreas
    [150,120, 90],  # right adrenal gland
    [153,153,153],  # left adrenal gland
    [  0,139,139],  # duodenum
    [250,170, 30],  # bladder
    [220,220,  0],  # prostate/uterus
])


def overlay_seg_anchor_sampling(seg, selected_indices):
    # seg: argmax label, shape: 2D (H, W), type: numpy.nd_array
    anatomical_structure_label = ["spleen", "right kidney", "left kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "inferior vena cava", "pancreas", "right adrenal gland", "left adrenal gland", "duodenum", "bladder", "prostate/uterus"]
    
    central_indices = selected_indices[0]
    boundary_indices = selected_indices[1]
    
    seg_selected_pixels = np.zeros([seg.shape[0], seg.shape[1], 3]).reshape(-1, 3)
    seg_selected_pixels[central_indices] = [255, 0, 0]
    seg_selected_pixels[boundary_indices] = [232, 191, 255]
    seg_selected_pixels = seg_selected_pixels.reshape((seg.shape[0], seg.shape[1], 3))

    # get unique labels
    labels = np.unique(seg)

    # remove background
    labels = labels[labels != 0]

    # final_image, here create a blank plot, then add plots inside the rois.
    final_img = np.zeros([seg.shape[0], seg.shape[1], 3])

    for l in labels:
        mask = seg == l
        img_f = seg_selected_pixels*mask[:, :, None]

        # # convert to rgb
        # img_f = img_f.transpose(1, 2, 0)

        # colored segmentation
        img_seg = colors[(l*mask).astype(int)]

        # alpha overlay
        final_img += 0.5*img_f + 0.5*img_seg
    
    return final_img


def overlay_img_seg_duodenum(img_slice, seg_mask):
    """
    img_slice: 2D (H, W), type: numpy.nd_array
    seg_mask: 2D (H, W), type: numpy.nd_array
    """
    duodenum_cls_id = 13
    
    seg_mask_cp = seg_mask.copy()
    seg_mask_cp[seg_mask != duodenum_cls_id] = 0
    
    duodenum_overlay = colors[seg_mask_cp.astype(int)]
    print(duodenum_overlay.shape)
    
    img_slice = np.tile(img_slice[:, :, None], (1, 1, 3))
    
    img_final = np.zeros([img_slice.shape[0], img_slice.shape[1], 3])
    img_final = img_slice * 0.5 + duodenum_overlay * 0.5
    
    return img_final


def dtm_anchor_sampling(y_hat, distance_d=5):
    selected_embedding_indices_list = {}

    this_y_hat = y_hat
    this_classes = np.unique(this_y_hat)

    for cls_i, cls_id in enumerate(this_classes):
        if cls_id != 13:
            continue
        # compute distance transform map
        cur_cls_y_hat_mask_npy = (this_y_hat == cls_id)
        dtm = distance(cur_cls_y_hat_mask_npy)

        boundary_intensity = np.percentile(dtm[cur_cls_y_hat_mask_npy == 1], distance_d)
        print(boundary_intensity)
        # region_inside_intensity = np.percentile(dtm[cur_cls_y_hat_mask_npy == 1], 70)

        dtm_flatten = dtm.flatten()
        # distance_dict = {1: 5, 3: 8, 4: 8} if is_boundary else {1: 7, 3: 10, 4: 10}
        # boundary_indices = (dtm_flatten == 1).nonzero()[0]
        # inside_indices = (dtm_flatten >= 3).nonzero()[0]
        
        inside_indices = (dtm_flatten > boundary_intensity).nonzero()[0]
        boundary_indices = ((dtm_flatten <= boundary_intensity) & (dtm_flatten > 0)).nonzero()[0]

        # num_inside = inside_indices.shape[0]
        # num_boundary = boundary_indices.shape[0]
        # n_indices = num_inside + num_boundary

        # boundary_indices = boundary_indices[::2]
        # inside_indices = inside_indices[::3]

        # inside_indices = np.random.choice(inside_indices, size=num_indside_sel, replace=False)
        # boundary_indices = np.random.choice(boundary_indices, size=num_boundary_sel, replace=False)
        indices = np.concatenate([inside_indices, boundary_indices])
        indices = [inside_indices, boundary_indices]

        selected_embedding_indices_list[cls_id] = indices

    return selected_embedding_indices_list, boundary_intensity


def crop_bbox_2d(nonzero_mask, cls_id):
    outside_value = 0
    mask_voxel_coords = np.where(nonzero_mask == cls_id)
    # minzidx = int(np.min(mask_voxel_coords[0]))
    # maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    # minxidx = int(np.min(mask_voxel_coords[1]))
    # maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    # bbox = [[minzidx, maxzidx], [minxidx, maxxidx]]
    # resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]))
    # return resizer

    # we find distance map wont treat image boundary as object boundary
    # so we crop a little more outside to cover the object boundary
    minzidx = int(np.min(mask_voxel_coords[0])) - 10
    maxzidx = int(np.max(mask_voxel_coords[0])) + 10
    minxidx = int(np.min(mask_voxel_coords[1])) - 10
    maxxidx = int(np.max(mask_voxel_coords[1])) + 10
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx]]
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]))
    return resizer


def visualize_duodenum_slice(mask, slice_num, distance_d):
    cls_id = 13
    resizer = crop_bbox_2d(mask, cls_id=cls_id)
    cropped_mask = mask[resizer]
    
    selected_pixel_indices, distance_threshold = dtm_anchor_sampling(cropped_mask, distance_d=distance_d)
    duodenum_pixel_indices = selected_pixel_indices[cls_id]
    
    label_cp = cropped_mask.copy()
    label_cp[cropped_mask != cls_id] = 0

    cls_id_overlay = overlay_seg_anchor_sampling(label_cp.astype(int), duodenum_pixel_indices)

    img = Image.fromarray(cls_id_overlay.astype(np.uint8))
    # .resize((256, 256), Image.ANTIALIAS)

    img.save(os.path.join('data', 'anchors_overlay_{}_distancePercentile_{}_threshold_{}.png'.format(int(slice_num), distance_d, distance_threshold)))
    pass


def visualize_img_duodenum_overlay(img_slice, seg_mask, slice_num):
    img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice)) * 255.0
    duodenum_overlay = overlay_img_seg_duodenum(img_slice, seg_mask)

    img = Image.fromarray(duodenum_overlay.astype(np.uint8))
    img.save(os.path.join('data', 'duodenum_overlay_{}.png'.format(slice_num)))
    
    resizer = crop_bbox_2d(seg_mask, cls_id=13)
    cropped_duodenum_overlay = duodenum_overlay[resizer]
    img = Image.fromarray(cropped_duodenum_overlay.astype(np.uint8))
    img.save(os.path.join('data', 'cropped_duodenum_overlay_{}.png'.format(slice_num)))


# visualization abdomen duodenum: 56 (small) & 61 (irregular & two independent parts)
downsample_size = (256, 256)
seg_nii_path = "data/amos_0001_label.nii.gz"
seg = nib.load(seg_nii_path).get_fdata()
seg = seg.transpose((1, 0, 2))[::-1, :, :][200:600, 146:678, :]
img_nii_path = "data/amos_0001.nii.gz"
img = nib.load(img_nii_path).get_fdata()
img = np.clip(img, -10, 60).transpose((1, 0, 2))[::-1, :, :][200:600, 146:678, :]
# for i in range(img.shape[2]):
#     if i == 58:
#         img_slice = img[:, :, i]
#         img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice)) * 255.0
#         cv2.imwrite("data/img_slice_{}.png".format(i), img_slice)

# small_narrow_duodenum_mask = seg[:, :, 55]
# irregular_duodenum_mask = seg[:, :, 60]
slice_num = 55
duodenum_mask = seg[:, :, slice_num]
distances_d = [5, 30, 50]
for d in distances_d:
    visualize_duodenum_slice(duodenum_mask, slice_num, d)

visualize_img_duodenum_overlay(img[:, :, slice_num], duodenum_mask, slice_num)




# data_npz = np.load(data_npz_path)
# slice = data_npz["slice"]
# label = data_npz["label"].transpose((2, 0, 1))
# label_numpy = torch.argmax(torch.from_numpy(label), dim=0).numpy()
# print(label_numpy.shape, np.unique(label_numpy))
# label = F.interpolate(torch.argmax(torch.from_numpy(label), dim=0).unsqueeze(0).unsqueeze(0).float(), size=downsample_size, mode="nearest")
# label = torch.squeeze(label).numpy()
# print(label.shape, np.unique(label))

# if "mr" in data_npz_path:
#     # for MR we predict its label for visualization beacause of incorrectness of original label
#     from models.seg_model import DilateResUNetCLMem, ResUNetDecoder

#     # --------------------
#     # Construct Networks only for MR prediction
#     # --------------------
#     pretrain_model_path = "runs/complete_ablation/runs/config_mr2ct/ITDFN+D_p+CCFA+CR/55389/models/23999"
#     seg_net = DilateResUNetCLMem(n_channels=1, n_classes=5, norm="InstanceNorm", act="leaky_relu", latent_dims=128)
#     seg_net.load_state_dict(torch.load(os.path.join(pretrain_model_path, "base_model_23999.pt")))
#     seg_net = seg_net.cuda()

#     param1 = -1.8
#     param2 = 4.4

#     slice = 2.0*(slice - param1)/(param2 - param1) - 1.0

#     out_dict = seg_net(torch.from_numpy(slice).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0).cuda())
#     label = torch.argmax(torch.softmax(out_dict['seg'], dim=1), dim=1).data.cpu().squeeze()

#     label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=downsample_size, mode="nearest").squeeze().numpy()


# # label = F.interpolate(torch.from_numpy(label))
# selected_1d_clsid_indices_dict = dtm_anchor_sampling(label, True if "mr" in data_npz_path else False)
# # selected_1d_clsid_indices_dict = dtm_anchor_sampling(label, True)

# save_overlay_dir = "runs/selected_anchors_visulization/{}".format("mr" if "mr" in data_npz_path else "ct")
# os.makedirs(save_overlay_dir, exist_ok=True)
# for cls_id, cls_indices in selected_1d_clsid_indices_dict.items():
#     label_cp = label.copy()
#     label_cp[label != cls_id] = 0

#     print(len(cls_indices), cls_indices[:10])

#     cls_id_overlay = overlay_seg_anchor_sampling(label_cp.astype(np.int), cls_indices)

#     img = Image.fromarray(cls_id_overlay.astype(np.uint8))
#     # .resize((256, 256), Image.ANTIALIAS)

#     img.save(os.path.join(save_overlay_dir, 'anchors_overlay_{}.png'.format(int(cls_id))))