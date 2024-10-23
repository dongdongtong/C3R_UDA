import os
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

np.random.seed(88)
torch.manual_seed(88)

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
    labels = labels[labels != 0]

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
        img_seg = colors[(l*mask).astype(np.int)]

        # alpha overlay
        final_img += 0.5*img_f + 0.5*img_seg
    
    return final_img


def dtm_anchor_sampling(y_hat, is_boundary=True):
    selected_embedding_indices_list = {}

    this_y_hat = y_hat
    this_classes = np.unique(this_y_hat)

    for cls_i, cls_id in enumerate(this_classes):
        if cls_id == 0:
            continue
        # compute distance transform map
        cur_cls_y_hat_mask_npy = (this_y_hat == cls_id)
        dtm = distance(cur_cls_y_hat_mask_npy)

        boundary_intensity = np.percentile(dtm[cur_cls_y_hat_mask_npy == 1], 1)
        region_inside_intensity = np.percentile(dtm[cur_cls_y_hat_mask_npy == 1], 70)

        dtm_flatten = dtm.flatten()
        distance_dict = {1: 5, 3: 8, 4: 8} if is_boundary else {1: 7, 3: 10, 4: 10}
        boundary_indices = (dtm_flatten == 1).nonzero()[0]
        inside_indices = (dtm_flatten >= distance_dict[int(cls_id)]).nonzero()[0]
        # boundary_indices = ((dtm_flatten <= boundary_intensity) & (dtm_flatten > 0)).nonzero()[0]

        num_inside = inside_indices.shape[0]
        num_boundary = boundary_indices.shape[0]
        n_indices = num_inside + num_boundary

        max_views = 20
        n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

        boundary_indices = boundary_indices[::2]
        inside_indices = inside_indices[::3]

        # if num_inside >= n_sel_pixels_clas_id / 2 and num_boundary >= n_sel_pixels_clas_id / 2:
        #     num_indside_sel = n_sel_pixels_clas_id // 2
        #     num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
        # elif num_inside >= n_sel_pixels_clas_id / 2:
        #     num_boundary_sel = num_boundary
        #     num_indside_sel = n_sel_pixels_clas_id - num_boundary_sel
        # elif num_boundary >= n_sel_pixels_clas_id / 2:
        #     num_indside_sel = num_inside
        #     num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
        # else:
        #     print('this shoud be never touched! {} {} {}'.format(num_inside, num_boundary, n_sel_pixels_clas_id))
        #     raise Exception

        # inside_indices = np.random.choice(inside_indices, size=num_indside_sel, replace=False)
        # boundary_indices = np.random.choice(boundary_indices, size=num_boundary_sel, replace=False)
        indices = np.concatenate([inside_indices, boundary_indices]) if is_boundary else inside_indices

        selected_embedding_indices_list[cls_id] = indices

    return selected_embedding_indices_list


# visualization validation set: mr 240 index, npz 888; ct 20/184 index, npz 68
downsample_size = (256, 256)
data_npz_path = "data/validation_ct/68.npz"
# data_npz_path = "data/validation_mr/888.npz"

data_npz = np.load(data_npz_path)
slice = data_npz["slice"]
label = data_npz["label"].transpose((2, 0, 1))
label_numpy = torch.argmax(torch.from_numpy(label), dim=0).numpy()
print(label_numpy.shape, np.unique(label_numpy))
label = F.interpolate(torch.argmax(torch.from_numpy(label), dim=0).unsqueeze(0).unsqueeze(0).float(), size=downsample_size, mode="nearest")
label = torch.squeeze(label).numpy()
print(label.shape, np.unique(label))

if "mr" in data_npz_path:
    # for MR we predict its label for visualization beacause of incorrectness of original label
    from models.seg_model import DilateResUNetCLMem, ResUNetDecoder

    # --------------------
    # Construct Networks only for MR prediction
    # --------------------
    pretrain_model_path = "runs/complete_ablation/runs/config_mr2ct/ITDFN+D_p+CCFA+CR/55389/models/23999"
    seg_net = DilateResUNetCLMem(n_channels=1, n_classes=5, norm="InstanceNorm", act="leaky_relu", latent_dims=128)
    seg_net.load_state_dict(torch.load(os.path.join(pretrain_model_path, "base_model_23999.pt")))
    seg_net = seg_net.cuda()

    param1 = -1.8
    param2 = 4.4

    slice = 2.0*(slice - param1)/(param2 - param1) - 1.0

    out_dict = seg_net(torch.from_numpy(slice).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0).cuda())
    label = torch.argmax(torch.softmax(out_dict['seg'], dim=1), dim=1).data.cpu().squeeze()

    label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=downsample_size, mode="nearest").squeeze().numpy()


# label = F.interpolate(torch.from_numpy(label))
selected_1d_clsid_indices_dict = dtm_anchor_sampling(label, True if "mr" in data_npz_path else False)
# selected_1d_clsid_indices_dict = dtm_anchor_sampling(label, True)

save_overlay_dir = "runs/selected_anchors_visulization/{}".format("mr" if "mr" in data_npz_path else "ct")
os.makedirs(save_overlay_dir, exist_ok=True)
for cls_id, cls_indices in selected_1d_clsid_indices_dict.items():
    label_cp = label.copy()
    label_cp[label != cls_id] = 0

    print(len(cls_indices), cls_indices[:10])

    cls_id_overlay = overlay_seg_anchor_sampling(label_cp.astype(np.int), cls_indices)

    img = Image.fromarray(cls_id_overlay.astype(np.uint8))
    # .resize((256, 256), Image.ANTIALIAS)

    img.save(os.path.join(save_overlay_dir, 'anchors_overlay_{}.png'.format(int(cls_id))))