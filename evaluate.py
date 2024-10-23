"""Code for testing SIFA."""
import json
import numpy as np
import os
import medpy.metric.binary as mmb

import time
import torch
from tqdm import tqdm

# data_size = [256, 256, 1]
# label_size = [256, 256, 1]
# data_size = [200, 200, 1]
# label_size = [200, 200, 1]

contour_map = {
    "bg": 0,
    "la_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4,
}


def read_lists(fid):
        """read test file list """

        with open(fid, 'r') as fd:
            _list = fd.readlines()

        my_list = []
        for _item in _list:
            my_list.append('data' + '/' + _item.split('\n')[0])
        print("read_lists:", my_list)
        return my_list

def label_decomp(label_batch):
        """decompose label for one-hot encoding """

        # input label_batch shape (batch_size, label_size, label_size)
        _batch_shape = list(label_batch.shape)
        _vol = np.zeros(_batch_shape)
        _vol[label_batch == 0] = 1
        _vol = _vol[..., np.newaxis]
        for i in range(5):
            if i == 0:
                continue
            _n_slice = np.zeros(label_batch.shape)
            _n_slice[label_batch == i] = 1
            _vol = np.concatenate( (_vol, _n_slice[..., np.newaxis]), axis=3)
        return np.float32(_vol)

def evaluate_mr(model, config=None, use_assd=False):
    """Test Function."""

    test_modality = "mr"
    # test_modality = "ct"
    test_dir = 'data/datalist/test_mr.txt'
    test_list = read_lists(test_dir)
    batch_size = 13

    dice_list = []
    assd_list = []
    for idx_file, fid in enumerate(test_list):

        _npz_dict = np.load(fid)

        if test_modality == "ct":
            label = _npz_dict['arr_0'].astype(np.float32)
            data = _npz_dict['arr_1'].astype(np.float32)
        else:
            label = _npz_dict['subject'].astype(np.float32)
            data = _npz_dict['label'].astype(np.float32)

        print(data.dtype, data.shape, data.min(), data.max())
        print(label.dtype, label.shape, label.min(), label.max())

        # This is to make the orientation of test data match with the training data
        # Set to False if the orientation of test data has already been aligned with the training data
        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)
        
        tmp_pred = np.zeros(label.shape)

        frame_list = [kk for kk in range(data.shape[2])]
        for ii in range(int(np.floor(data.shape[2] // batch_size))):
            data_batch = np.zeros([batch_size, 256, 256, 1])
            label_batch = np.zeros([batch_size, 256, 256])
            for idx, jj in enumerate(frame_list[ii * batch_size: (ii + 1) * batch_size]):
                # print(data_batch[idx, ...].shape, data[..., jj].copy().shape)
                data_batch[idx, ...] = np.expand_dims(data[..., jj].copy(), 2)
                label_batch[idx, ...] = label[..., jj].copy()  # return shape (batch_size, label_size, label_size, num_cls)
            
            label_batch = label_decomp(label_batch)

            if test_modality=='ct':
                data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -2.8), np.subtract(3.2, -2.8)), 2.0),1) # {-2.8, 3.2} need to be changed according to the data statistics
            elif test_modality=='mr':
                data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -1.8), np.subtract(4.4, -1.8)), 2.0),1)  # {-1.8, 4.4} need to be changed according to the data statistics
            # # data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -1.6), np.subtract(3.4, -1.6)), 2.0), 1)  # U -3.6061754 9.745765
            

            input_batch = torch.from_numpy(data_batch).permute(0,3,1,2).type(torch.FloatTensor).cuda()

            # outputs = self.segmenter_ult(self.encoder(input_batch)[0])
            # evaluate_latent_x1, evaluate_latent_x2, evaluate_latent_x3, evaluate_latent_x4, evaluate_latent_x5 = self.encoder(input_batch)
            # outputs, _ = self.segmenter_ult(x1=evaluate_latent_x1, x2=evaluate_latent_x2, x3=evaluate_latent_x3, x4=evaluate_latent_x4, x5=evaluate_latent_x5)
            # mr_multi_lv_feats = model.enc_img_toT_forward(input_batch)
            # fake_ct = model.dec_img_toT_forward(mr_multi_lv_feats)
            outputs = model(input_batch)
            compact_pred_outputs = torch.argmax(torch.softmax(outputs['seg'].permute(0,2,3,1), dim=-1), dim=-1).data.cpu().numpy()

            for idx, jj in enumerate(frame_list[ii * batch_size: (ii + 1) * batch_size]):
                tmp_pred[..., jj] = compact_pred_outputs[idx, ...].copy()

        for c in range(1, 5):
            pred_test_data_tr = tmp_pred.copy()
            pred_test_data_tr[pred_test_data_tr != c] = 0

            pred_gt_data_tr = label.copy()
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            dice = mmb.dc(pred_test_data_tr, pred_gt_data_tr)
            print(dice)
            dice_list.append(dice)
            if use_assd:
                assd = mmb.assd(pred_test_data_tr, pred_gt_data_tr)
                print(assd)
                assd_list.append(assd)

    dice_arr = 100 * np.reshape(dice_list, [4, -1]).transpose()

    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)

    print ('Dice:')
    print ('AA :%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
    print ('LAC:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
    print ('LVC:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
    print ('Myo:%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
    print ('Dice Mean:%.2f' % np.mean(dice_mean))

    if use_assd:
        assd_arr = np.reshape(assd_list, [4, -1]).transpose()

        assd_mean = np.mean(assd_arr, axis=1)
        assd_std = np.std(assd_arr, axis=1)

        print ('ASSD:')
        print ('AA :%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
        print ('LAC:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
        print ('LVC:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
        print ('Myo:%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
        print ('ASSD Mean:%.2f' % np.mean(assd_mean))

    return np.mean(dice_mean)


def evaluate_ct(model, config=None, use_assd=False):
    """Test Function."""

    test_modality = "ct"
    # test_modality = "ct"
    # test_dir = config['data'][test_modality]["test_list"]
    test_dir = 'data/datalist/test_ct.txt'
    test_list = read_lists(test_dir)
    batch_size = 16

    dice_list = []
    assd_list = []
    for idx_file, fid in enumerate(test_list):

        _npz_dict = np.load(fid)

        if test_modality == "ct":
            label = _npz_dict['arr_0'].astype(np.float32)
            data = _npz_dict['arr_1'].astype(np.float32)
        else:
            label = _npz_dict['subject'].astype(np.float32)
            data = _npz_dict['label'].astype(np.float32)

        print(data.dtype, data.shape, data.min(), data.max())
        print(label.dtype, label.shape, label.min(), label.max())

        # This is to make the orientation of test data match with the training data
        # Set to False if the orientation of test data has already been aligned with the training data
        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)
        
        tmp_pred = np.zeros(label.shape)

        frame_list = [kk for kk in range(data.shape[2])]
        for ii in range(int(np.floor(data.shape[2] // batch_size))):
            data_batch = np.zeros([batch_size, 256, 256, 1])
            label_batch = np.zeros([batch_size, 256, 256])
            for idx, jj in enumerate(frame_list[ii * batch_size: (ii + 1) * batch_size]):
                # print(data_batch[idx, ...].shape, data[..., jj].copy().shape)
                data_batch[idx, ...] = np.expand_dims(data[..., jj].copy(), 2)
                label_batch[idx, ...] = label[..., jj].copy()  # return shape (batch_size, label_size, label_size, num_cls)
            
            label_batch = label_decomp(label_batch)

            if test_modality=='ct':
                data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -2.8), np.subtract(3.2, -2.8)), 2.0),1) # {-2.8, 3.2} need to be changed according to the data statistics
            elif test_modality=='mr':
                data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -1.8), np.subtract(4.4, -1.8)), 2.0),1)  # {-1.8, 4.4} need to be changed according to the data statistics
            # # data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -1.6), np.subtract(3.4, -1.6)), 2.0), 1)  # U -3.6061754 9.745765
            

            input_batch = torch.from_numpy(data_batch).permute(0,3,1,2).type(torch.FloatTensor).cuda()

            # outputs = self.segmenter_ult(self.encoder(input_batch)[0])
            # evaluate_latent_x1, evaluate_latent_x2, evaluate_latent_x3, evaluate_latent_x4, evaluate_latent_x5 = self.encoder(input_batch)
            # outputs, _ = self.segmenter_ult(x1=evaluate_latent_x1, x2=evaluate_latent_x2, x3=evaluate_latent_x3, x4=evaluate_latent_x4, x5=evaluate_latent_x5)
            outputs = model(input_batch)
            # outputs = model(input_batch, use_ds=False)
            compact_pred_outputs = torch.argmax(torch.softmax(outputs['seg'].permute(0,2,3,1), dim=-1), dim=-1).data.cpu().numpy()

            for idx, jj in enumerate(frame_list[ii * batch_size: (ii + 1) * batch_size]):
                tmp_pred[..., jj] = compact_pred_outputs[idx, ...].copy()

        for c in range(1, 5):
            pred_test_data_tr = tmp_pred.copy()
            pred_test_data_tr[pred_test_data_tr != c] = 0

            pred_gt_data_tr = label.copy()
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            dice = mmb.dc(pred_test_data_tr, pred_gt_data_tr)
            print(dice)
            dice_list.append(dice)
            if use_assd:
                assd = mmb.assd(pred_test_data_tr, pred_gt_data_tr)
                print(assd)
                assd_list.append(assd)

    dice_arr = 100 * np.reshape(dice_list, [4, -1]).transpose()

    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)

    print ('Dice:')
    print ('AA :%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
    print ('LAC:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
    print ('LVC:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
    print ('Myo:%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
    print ('Dice Mean:%.2f' % np.mean(dice_mean))

    if use_assd:
        assd_arr = np.reshape(assd_list, [4, -1]).transpose()

        assd_mean = np.mean(assd_arr, axis=1)
        assd_std = np.std(assd_arr, axis=1)

        print ('ASSD:')
        print ('AA :%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
        print ('LAC:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
        print ('LVC:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
        print ('Myo:%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
        print ('ASSD Mean:%.2f' % np.mean(assd_mean))

    return np.mean(dice_mean)


def evaluate_ct_val(model, config=None, val_ct_dataloader=None, use_assd=False):
    print("evaluate_ct_val visulizing all slice as a whole image")
    all_pred_mask = []
    all_label = []
    
    with torch.no_grad():
        for item in tqdm(val_ct_dataloader):
            img, label, _ = item
            img = img.cuda()
            out_dict = model(img)
            all_pred_mask.append(torch.argmax(torch.softmax(out_dict['seg'], dim=1), dim=1).cpu().numpy())  # shape (B, H, W)
            all_label.append(np.argmax(label, axis=-1))  # shape (B, H, W)
    all_pred_mask = np.concatenate(all_pred_mask, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    # calculate dice on the whole pred mask using mmb.dc
    dice_list = []
    assd_list = []
    for c in range(1, 5):
        pred_test_data_tr = all_pred_mask.copy()
        pred_test_data_tr[pred_test_data_tr != c] = 0

        pred_gt_data_tr = all_label.copy()
        pred_gt_data_tr[pred_gt_data_tr != c] = 0

        dice = mmb.dc(pred_test_data_tr, pred_gt_data_tr)
        print(dice)
        dice_list.append(dice)
        if use_assd:
            assd = mmb.assd(pred_test_data_tr, pred_gt_data_tr)
            print(assd)
            assd_list.append(assd)
    
    dice_list = 100 * np.array(dice_list)
    print()
    print ('Validation CT Dice:')
    print ('AA :%.2f' % (dice_list[3]))
    print ('LAC:%.2f' % (dice_list[1]))
    print ('LVC:%.2f' % (dice_list[2]))
    print ('Myo:%.2f' % (dice_list[0]))
    print ('Dice Mean:%.2f' % np.mean(dice_list))
    del all_pred_mask
    del all_label

    return np.mean(dice_list)

def evaluate_mr_val(model, config=None, val_mr_dataloader=None, use_assd=False):
    print("evaluate_mr_val visulizing all slice as a whole image")
    all_pred_mask = []
    all_label = []
    
    with torch.no_grad():
        for item in tqdm(val_mr_dataloader):
            img, label, _ = item
            img = img.cuda()
            out_dict = model(img)
            all_pred_mask.append(torch.argmax(torch.softmax(out_dict['seg'], dim=1), dim=1).cpu().numpy())  # shape (B, H, W)
            all_label.append(np.argmax(label, axis=-1))  # shape (B, H, W)
    all_pred_mask = np.concatenate(all_pred_mask, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    # calculate dice on the whole pred mask using mmb.dc
    dice_list = []
    assd_list = []
    for c in range(1, 5):
        pred_test_data_tr = all_pred_mask.copy()
        pred_test_data_tr[pred_test_data_tr != c] = 0

        pred_gt_data_tr = all_label.copy()
        pred_gt_data_tr[pred_gt_data_tr != c] = 0

        dice = mmb.dc(pred_test_data_tr, pred_gt_data_tr)
        print(dice)
        dice_list.append(dice)
        if use_assd:
            assd = mmb.assd(pred_test_data_tr, pred_gt_data_tr)
            print(assd)
            assd_list.append(assd)
    
    dice_list = 100 * np.array(dice_list)
    print()
    print ('Validation CT Dice:')
    print ('AA :%.2f' % (dice_list[3]))
    print ('LAC:%.2f' % (dice_list[1]))
    print ('LVC:%.2f' % (dice_list[2]))
    print ('Myo:%.2f' % (dice_list[0]))
    print ('Dice Mean:%.2f' % np.mean(dice_list))
    del all_pred_mask
    del all_label

    return np.mean(dice_list)

