import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import timeit

import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

class PixelContrastLoss(nn.Module):
    def __init__(self, configer, writer, visual):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.writer = writer
        self.visual = visual

        self.temperature = self.configer['training']['temperature']
        self.base_temperature = self.configer['training']['base_temperature']

        self.ignore_label = 5
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        self.max_samples = self.configer['training']['max_samples']
        self.max_views = self.configer['training']['max_views']
        self.max_views_bg = self.configer['training']['max_views_bg']
        self.dtm_perc = self.configer['training']['dtm_perc']
        self.target_dtm_perc = self.configer['training']['target_dtm_perc']
        self.target_max_views_bg = self.configer['training']['target_max_views_bg']
        self.target_max_views = self.configer['training']['target_max_views']

        self.pixel_memory_size = configer['training']["pixel_memory_size"]
        self.segment_memory_size = configer['training']["segment_memory_size"]

    def _dtm_anchor_sampling(self, X, y_hat):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        X_ = None
        y_ = None

        selected_embedding_indices_list = []

        if torch.unique(y_hat).shape[0] == 1:
            return X_, y_, selected_embedding_indices_list
        else:
            for ii in range(batch_size):
                this_y_hat = y_hat[ii]
                this_classes = torch.unique(this_y_hat)
                this_classes = [x for x in this_classes if x != self.ignore_label]
                this_classes = [x for x in this_classes if (this_y_hat == x).nonzero().shape[0] > 10]

                selected_embedding_indices_ = None
                for cls_i, cls_id in enumerate(this_classes):
                    cur_cls_y_hat_mask = (this_y_hat == cls_id)

                    # compute distance transform map
                    cur_cls_y_hat_mask_npy = cur_cls_y_hat_mask.cpu().numpy()
                    dtm = distance(cur_cls_y_hat_mask_npy)
                    in_out_perc_intensity = np.percentile(dtm[cur_cls_y_hat_mask_npy == 1], self.dtm_perc)

                    dtm_flatten = dtm.flatten()
                    inside_indices = (dtm_flatten > in_out_perc_intensity).nonzero()[0]
                    boundary_indices = ((dtm_flatten <= in_out_perc_intensity) & (dtm_flatten > 0)).nonzero()[0]

                    num_inside = inside_indices.shape[0]
                    num_boundary = boundary_indices.shape[0]
                    n_indices = num_inside + num_boundary

                    max_views = self.max_views_bg if cls_id == 0 else self.max_views
                    n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

                    if num_inside >= n_sel_pixels_clas_id / 2 and num_boundary >= n_sel_pixels_clas_id / 2:
                        num_indside_sel = n_sel_pixels_clas_id // 2
                        num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
                    elif num_inside >= n_sel_pixels_clas_id / 2:
                        num_boundary_sel = num_boundary
                        num_indside_sel = n_sel_pixels_clas_id - num_boundary_sel
                    elif num_boundary >= n_sel_pixels_clas_id / 2:
                        num_indside_sel = num_inside
                        num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
                    else:
                        print('this shoud be never touched! {} {} {}'.format(num_inside, num_boundary, n_sel_pixels_clas_id))
                        raise Exception

                    # perm = torch.randperm(num_inside)
                    # inside_indices = inside_indices[perm[:num_indside_sel]]

                    # perm = torch.randperm(num_boundary)
                    # boundary_indices = boundary_indices[perm[:num_boundary_sel]]
                    # indices = torch.cat((inside_indices, boundary_indices), dim=0)

                    inside_indices = np.random.choice(inside_indices, size=num_indside_sel, replace=False)
                    boundary_indices = np.random.choice(boundary_indices, size=num_boundary_sel, replace=False)
                    indices = np.concatenate([inside_indices, boundary_indices])

                    if indices.shape[0] < 2:
                        continue

                    if type(selected_embedding_indices_) == type(None):
                        selected_embedding_indices_ = indices
                    else:
                        selected_embedding_indices_ = np.concatenate([selected_embedding_indices_, indices])

                    if type(X_) == type(None) and type(y_) == type(None):
                        X_ = X[ii, indices, :].squeeze()
                        y_ = torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device=X.device).fill_(cls_id)
                    else:
                        X_ = torch.cat([X_, X[ii, indices, :].squeeze()], dim=0)
                        y_ = torch.cat([y_, torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device=X.device).fill_(cls_id)])

                selected_embedding_indices_list.append(selected_embedding_indices_)

        return X_, y_, selected_embedding_indices_list

    def _dtm_target_anchor_sampling(self, X, y_hat, boundary=False):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        X_ = None
        y_ = None

        selected_embedding_indices_list = []

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_classes = torch.unique(this_y_hat)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y_hat == x).nonzero().shape[0] > 10]
            # print()
            # print("-----------------------------------")
            # print("this_classes values", this_classes)

            selected_embedding_indices_ = None
            for cls_i, cls_id in enumerate(this_classes):
                cur_cls_y_hat_mask = (this_y_hat == cls_id)

                # compute distance transform map
                cur_cls_y_hat_mask_npy = cur_cls_y_hat_mask.cpu().numpy()
                dtm = distance(cur_cls_y_hat_mask_npy)
                in_out_perc_intensity = np.percentile(dtm[cur_cls_y_hat_mask_npy == 1], self.target_dtm_perc)

                dtm_flatten = dtm.flatten()
                inside_indices = (dtm_flatten > in_out_perc_intensity).nonzero()[0]
                boundary_indices = ((dtm_flatten <= in_out_perc_intensity) & (dtm_flatten > 0)).nonzero()[0]

                num_inside = inside_indices.shape[0]
                num_boundary = boundary_indices.shape[0]
                n_indices = num_inside + num_boundary

                max_views = self.target_max_views_bg if cls_id == 0 else self.target_max_views
                n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

                if num_inside >= n_sel_pixels_clas_id / 2 and num_boundary >= n_sel_pixels_clas_id / 2:
                    num_indside_sel = n_sel_pixels_clas_id // 2
                    num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
                elif num_inside >= n_sel_pixels_clas_id / 2:
                    num_boundary_sel = num_boundary
                    num_indside_sel = n_sel_pixels_clas_id - num_boundary_sel
                elif num_boundary >= n_sel_pixels_clas_id / 2:
                    num_indside_sel = num_inside
                    num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_inside, num_boundary, n_sel_pixels_clas_id))
                    raise Exception

                # perm = torch.randperm(num_inside)
                # inside_indices = inside_indices[perm[:num_indside_sel]]

                # perm = torch.randperm(num_boundary)
                # boundary_indices = boundary_indices[perm[:num_boundary_sel]]
                # indices = torch.cat((inside_indices, boundary_indices), dim=0)

                inside_indices = np.random.choice(inside_indices, size=num_indside_sel, replace=False)
                boundary_indices = np.random.choice(boundary_indices, size=num_boundary_sel, replace=False)
                indices = np.concatenate([inside_indices, boundary_indices]) if boundary else inside_indices

                # print("this_classes element label cls_id", cls_id, "indices.shape[0]:", indices.shape[0])

                if indices.shape[0] < 2:
                    continue

                if type(selected_embedding_indices_) == type(None):
                    selected_embedding_indices_ = indices
                else:
                    selected_embedding_indices_ = np.concatenate([selected_embedding_indices_, indices])

                if type(X_) == type(None) and type(y_) == type(None):
                    temp_X = X[ii, indices, :].squeeze(1)
                    X_ = temp_X
                    y_ = torch.zeros(temp_X.shape[0], dtype=torch.long, device="cuda:0").fill_(cls_id)
                else:
                    temp_X = X[ii, indices, :].squeeze(1)
                    X_ = torch.cat([X_, temp_X], dim=0)
                    y_ = torch.cat([y_, torch.zeros(temp_X.shape[0], dtype=torch.long, device="cuda:0").fill_(cls_id)])
            
            selected_embedding_indices_list.append(selected_embedding_indices_)

        return X_, y_, selected_embedding_indices_list

    def _pixel_contrastive(self, X_anchor, y_anchor, iter=None, cl_lv="sr"):
        # print()
        # print("-------------------------------------")
        # print("X_anchor NAN?", torch.isnan(X_anchor).sum())
        anchor_num = X_anchor.shape[0]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor

        contrast_feature = X_anchor.detach()
        
        mask = torch.eq(y_anchor, y_anchor.T).float().cuda()
        # print("-------------------------------------")
        # print("mask NAN?", torch.isnan(mask).sum())

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        # print("-------------------------------------")
        # print("anchor_dot_contrast NAN?", torch.isnan(anchor_dot_contrast).sum())

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast

        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        with torch.no_grad():
            if iter % self.configer['training']['print_pos_neg_hist'] == 0 or (iter + 1) == self.configer['training']['train_iters']:
                self.writer.add_histogram("contrast/{} pos values".format(cl_lv), logits[mask != 0], iter)
                self.writer.add_histogram("contrast/{} neg values".format(cl_lv), logits[neg_mask != 0], iter)
                # self.writer.add_histogram("contrast/selected anchors", y_anchor, iter)

        neg_logits = torch.exp(logits) * neg_mask
        # print("-------------------------------------")
        # print("neg_logits NAN?", torch.isnan(neg_logits).sum())
        neg_logits = neg_logits.sum(1, keepdim=True)
        # print("neg_logits.sum() NAN?", torch.isnan(neg_logits).sum())

        exp_logits = torch.exp(logits)
        # print("-------------------------------------")
        # print("exp_logits NAN?", torch.isnan(exp_logits).sum())

        log_prob = logits - torch.log(exp_logits + neg_logits)
        # print("-------------------------------------")
        # print("log_prob NAN?", torch.isnan(log_prob).sum())
        # log_prob = logits - torch.log(neg_logits)  # decoupled cl loss

        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        # print("-------------------------------------")
        # print("(mask * log_prob).sum(1) NAN?", torch.isnan((mask * log_prob).sum(1)).sum())
        # print("mask.sum(1) NAN?", torch.isnan(mask.sum(1)).sum())
        # print("(mask * log_prob).sum(1) / mask.sum(1) NAN?", torch.isnan((mask * log_prob).sum(1) / mask.sum(1)).sum())
        # print("mean_log_prob_pos NAN?", torch.isnan(mean_log_prob_pos).sum())
        if torch.isnan(mean_log_prob_pos).sum() > 0:
            print("mask.sum(1) value", mask.sum(1))
            zero_index = (mask.sum(1) == 0).nonzero().squeeze()
            print("zero_index", zero_index)
            print("zero_index label", y_anchor[zero_index])
            print("current y_anchor:", y_anchor)
            raise Exception

        loss = mean_log_prob_pos.mean()

        return loss

    def forward(self, source_feats, target_feats, source_labels=None, target_labels=None, iter=None, cl_lv="sr"):
        source_labels = source_labels.unsqueeze(1).float().clone()
        source_labels = torch.nn.functional.interpolate(source_labels,
                                                 (source_feats.shape[2], source_feats.shape[3]), mode='nearest')
        source_labels = source_labels.squeeze(1).long()
        assert source_labels.shape[-1] == source_feats.shape[-1], '{} {}'.format(source_labels.shape, source_feats.shape)

        target_labels = target_labels.unsqueeze(1).float().clone()
        target_labels = torch.nn.functional.interpolate(target_labels,
                                                 (target_feats.shape[2], target_feats.shape[3]), mode='nearest')
        target_labels = target_labels.squeeze(1).long()
        assert target_labels.shape[-1] == target_feats.shape[-1], '{} {}'.format(target_labels.shape, target_feats.shape)

        # labels = labels.contiguous().view(batch_size, -1)
        # predict = predict.contiguous().view(batch_size, -1)
        source_feats = source_feats.permute(0, 2, 3, 1)
        source_feats = source_feats.contiguous().view(source_feats.shape[0], -1, source_feats.shape[-1])  # shape B*(H*W)*FeatDim
        # feats_, labels_ = self._new_hard_anchor_sampling(feats, labels, predict)
        source_feats_, source_labels_, source_selected_embedding_indices_list = self._dtm_anchor_sampling(source_feats, source_labels)
        # print("source_feats_ shape", source_feats_.shape, "source_labels_ shape", source_labels_.shape)  # source_feats_ shape torch.Size([446, 128]) source_labels_ shape torch.Size([446])

        target_feats = target_feats.permute(0, 2, 3, 1)
        target_feats = target_feats.contiguous().view(target_feats.shape[0], -1, target_feats.shape[-1])  # shape B*(H*W)*FeatDim
        target_feats_, target_labels_, target_selected_embedding_indices_list = self._dtm_target_anchor_sampling(target_feats, target_labels)
        # print("target_feats_ shape", target_feats_.shape, "target_labels_ shape", target_labels_.shape)  # target_feats_ shape torch.Size([21, 128]) target_labels_ shape torch.Size([732])

        if type(source_feats_) == type(None) and type(target_feats_) == type(None):
            return torch.tensor(0.)
        elif type(source_feats_) == type(None):
            feats_ = target_feats_
            labels_ = target_labels_
            
            if iter % self.configer['training']['val_interval'] == 0:
            # if self.iter % 60 == 0:
                self.visual.display_anchor_sampling_on_seg(target_labels.cpu().numpy(), target_selected_embedding_indices_list, "t", iter, cl_lv=cl_lv)
        elif type(target_feats_) == type(None):
            feats_ = source_feats_
            labels_ = source_labels_

            if iter % self.configer['training']['val_interval'] == 0:
            # if self.iter % 60 == 0:
                self.visual.display_anchor_sampling_on_seg(source_labels.cpu().numpy(), source_selected_embedding_indices_list, "s", iter, cl_lv=cl_lv)
        else:
            feats_ = torch.cat([source_feats_, target_feats_])
            labels_ = torch.cat([source_labels_, target_labels_])
            if iter % self.configer['training']['val_interval'] == 0:
            # if self.iter % 60 == 0:
                self.visual.display_anchor_sampling_on_seg(source_labels.cpu().numpy(), source_selected_embedding_indices_list, "s", iter, cl_lv=cl_lv)
                self.visual.display_anchor_sampling_on_seg(target_labels.cpu().numpy(), target_selected_embedding_indices_list, "t", iter, cl_lv=cl_lv)

        # if type(source_feats_) == type(None) and type(source_labels_) == type(None):
        #     return torch.tensor(0.)

        loss = self._pixel_contrastive(feats_, labels_, iter=iter, cl_lv=cl_lv)

        return loss


class PixelContrastLossMem(nn.Module):
    def __init__(self, configer, writer, visual):
        super(PixelContrastLossMem, self).__init__()

        self.configer = configer
        self.writer = writer
        self.visual = visual

        self.temperature = self.configer['training']['temperature']
        self.base_temperature = self.configer['training']['base_temperature']

        self.ignore_label = 5
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        self.max_samples = self.configer['training']['max_samples']
        self.max_views = self.configer['training']['max_views']
        self.max_views_bg = self.configer['training']['max_views_bg']
        self.dtm_perc = self.configer['training']['dtm_perc']
        self.target_dtm_perc = self.configer['training']['target_dtm_perc']
        self.target_max_views_bg = self.configer['training']['target_max_views_bg']
        self.target_max_views = self.configer['training']['target_max_views']

        self.random_samples = self.configer['training']['random_samples']  # 8000, this is the pixels select from the memory bank

        self.pixel_memory_size = configer['training']["pixel_memory_size"]
        self.segment_memory_size = configer['training']["segment_memory_size"]

    def _dtm_anchor_sampling(self, X, y_hat):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        X_ = None
        y_ = None

        selected_embedding_indices_list = []

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_classes = torch.unique(this_y_hat)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y_hat == x).nonzero().shape[0] > 10]

            selected_embedding_indices_ = None
            for cls_i, cls_id in enumerate(this_classes):
                cur_cls_y_hat_mask = (this_y_hat == cls_id)

                # compute distance transform map
                cur_cls_y_hat_mask_npy = cur_cls_y_hat_mask.cpu().numpy()
                dtm = distance(cur_cls_y_hat_mask_npy)
                in_out_perc_intensity = np.percentile(dtm[cur_cls_y_hat_mask_npy == 1], self.dtm_perc)

                dtm_flatten = dtm.flatten()
                inside_indices = (dtm_flatten > in_out_perc_intensity).nonzero()[0]
                boundary_indices = ((dtm_flatten <= in_out_perc_intensity) & (dtm_flatten > 0)).nonzero()[0]

                num_inside = inside_indices.shape[0]
                num_boundary = boundary_indices.shape[0]
                n_indices = num_inside + num_boundary

                max_views = self.max_views_bg if cls_id == 0 else self.max_views
                n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

                if num_inside >= n_sel_pixels_clas_id / 2 and num_boundary >= n_sel_pixels_clas_id / 2:
                    num_indside_sel = n_sel_pixels_clas_id // 2
                    num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
                elif num_inside >= n_sel_pixels_clas_id / 2:
                    num_boundary_sel = num_boundary
                    num_indside_sel = n_sel_pixels_clas_id - num_boundary_sel
                elif num_boundary >= n_sel_pixels_clas_id / 2:
                    num_indside_sel = num_inside
                    num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_inside, num_boundary, n_sel_pixels_clas_id))
                    raise Exception

                # perm = torch.randperm(num_inside)
                # inside_indices = inside_indices[perm[:num_indside_sel]]

                # perm = torch.randperm(num_boundary)
                # boundary_indices = boundary_indices[perm[:num_boundary_sel]]
                # indices = torch.cat((inside_indices, boundary_indices), dim=0)

                inside_indices = np.random.choice(inside_indices, size=num_indside_sel, replace=False)
                boundary_indices = np.random.choice(boundary_indices, size=num_boundary_sel, replace=False)
                indices = np.concatenate([inside_indices, boundary_indices])


                if indices.shape[0] < 1:
                    continue

                if type(selected_embedding_indices_) == type(None):
                    selected_embedding_indices_ = indices
                else:
                    selected_embedding_indices_ = np.concatenate([selected_embedding_indices_, indices])

                if type(X_) == type(None) and type(y_) == type(None):
                    X_ = X[ii, indices, :].squeeze()
                    y_ = torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)
                else:
                    X_ = torch.cat([X_, X[ii, indices, :].squeeze()], dim=0)
                    y_ = torch.cat([y_, torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)])
            
            selected_embedding_indices_list.append(selected_embedding_indices_)

        return X_, y_, selected_embedding_indices_list

    def _dtm_target_anchor_sampling(self, X, y_hat, boundary=False):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        X_ = None
        y_ = None

        selected_embedding_indices_list = []

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_classes = torch.unique(this_y_hat)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y_hat == x).nonzero().shape[0] > 10]

            selected_embedding_indices_ = None
            for cls_i, cls_id in enumerate(this_classes):
                cur_cls_y_hat_mask = (this_y_hat == cls_id)

                # compute distance transform map
                cur_cls_y_hat_mask_npy = cur_cls_y_hat_mask.cpu().numpy()
                dtm = distance(cur_cls_y_hat_mask_npy)
                in_out_perc_intensity = np.percentile(dtm[cur_cls_y_hat_mask_npy == 1], self.target_dtm_perc)

                dtm_flatten = dtm.flatten()
                inside_indices = (dtm_flatten > in_out_perc_intensity).nonzero()[0]
                boundary_indices = ((dtm_flatten <= in_out_perc_intensity) & (dtm_flatten > 0)).nonzero()[0]

                num_inside = inside_indices.shape[0]
                num_boundary = boundary_indices.shape[0]
                n_indices = num_inside + num_boundary

                max_views = self.target_max_views_bg if cls_id == 0 else self.target_max_views
                n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

                if num_inside >= n_sel_pixels_clas_id / 2 and num_boundary >= n_sel_pixels_clas_id / 2:
                    num_indside_sel = n_sel_pixels_clas_id // 2
                    num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
                elif num_inside >= n_sel_pixels_clas_id / 2:
                    num_boundary_sel = num_boundary
                    num_indside_sel = n_sel_pixels_clas_id - num_boundary_sel
                elif num_boundary >= n_sel_pixels_clas_id / 2:
                    num_indside_sel = num_inside
                    num_boundary_sel = n_sel_pixels_clas_id - num_indside_sel
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_inside, num_boundary, n_sel_pixels_clas_id))
                    raise Exception

                # perm = torch.randperm(num_inside)
                # inside_indices = inside_indices[perm[:num_indside_sel]]

                # perm = torch.randperm(num_boundary)
                # boundary_indices = boundary_indices[perm[:num_boundary_sel]]
                # indices = torch.cat((inside_indices, boundary_indices), dim=0)

                inside_indices = np.random.choice(inside_indices, size=num_indside_sel, replace=False)
                boundary_indices = np.random.choice(boundary_indices, size=num_boundary_sel, replace=False)
                indices = np.concatenate([inside_indices, boundary_indices]) if boundary else inside_indices

                if indices.shape[0] < 2:
                    continue

                if type(selected_embedding_indices_) == type(None):
                    selected_embedding_indices_ = indices
                else:
                    selected_embedding_indices_ = np.concatenate([selected_embedding_indices_, indices])

                if type(X_) == type(None) and type(y_) == type(None):
                    temp_X = X[ii, indices, :].squeeze(1)
                    X_ = temp_X
                    y_ = torch.zeros(temp_X.shape[0], dtype=torch.long, device="cuda:0").fill_(cls_id)
                else:
                    temp_X = X[ii, indices, :].squeeze(1)
                    X_ = torch.cat([X_, temp_X], dim=0)
                    y_ = torch.cat([y_, torch.zeros(temp_X.shape[0], dtype=torch.long, device="cuda:0").fill_(cls_id)])
            
            selected_embedding_indices_list.append(selected_embedding_indices_)

        return X_, y_, selected_embedding_indices_list

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * self.random_samples, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * self.random_samples)).long().cuda()

        # indices = torch.tensor(random.sample(range(cache_size // 2), self.random_samples))
        # indices = torch.tensor(indices)
        # sampled_values = values[indices]
        
        pixel_perm = torch.randperm(self.pixel_memory_size)
        # segment_perm = torch.randperm(self.segment_memory_size)

        sample_ptr = 0
        for ii in range(class_num):
            # X_[sample_ptr:sample_ptr + self.random_samples, ...] = Q[ii, segment_perm[:self.random_samples], :]
            # X_[sample_ptr + self.random_samples:sample_ptr + self.random_samples * 2, ...] = Q[ii, pixel_perm[:self.random_samples] + self.segment_memory_size, :]
            # y_[sample_ptr:sample_ptr + self.random_samples * 2, ...] = ii
            # sample_ptr += self.random_samples * 2

            X_[sample_ptr:sample_ptr + self.random_samples, ...] = Q[ii, pixel_perm[:self.random_samples], :]
            y_[sample_ptr:sample_ptr + self.random_samples] = ii
            sample_ptr += self.random_samples

        return X_, y_
    
    def _contrastive(self, X_anchor, y_anchor, queue=None, iter=None, domain="s", cl_lv="sr"):
        anchor_num = X_anchor.shape[0]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor

        X_contrast, y_contrast = self._sample_negative(queue)
        # ema_y_anchor = ema_y_anchor.contiguous().view(-1, 1)
        contrast_feature = torch.cat([anchor_feature, X_contrast])
        y_contrast = torch.cat([y_anchor, y_contrast.contiguous().view(-1, 1)])
        
        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)

        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast

        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(mask.shape[0]).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        with torch.no_grad():
            if iter % self.configer['training']['print_pos_neg_hist'] == 0 or (iter + 1) == self.configer['training']['train_iters']:
                self.writer.add_histogram("contrast/{}_{} pos values".format(domain, cl_lv), logits[mask != 0], iter)
                self.writer.add_histogram("contrast/{}_{} neg values".format(domain, cl_lv), logits[neg_mask != 0], iter)
                # self.writer.add_histogram("contrast/{} selected anchors", y_contrast, iter)

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        # log_prob = logits - torch.log(neg_logits)  # decoupled cl loss

        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask.sum(1)

        loss = mean_log_prob_pos.mean()

        return loss

    def forward(self, source_feats, target_feats, source_labels=None, target_labels=None, source_queue=None, iter=None, cl_lv="sr"):
        source_labels = source_labels.unsqueeze(1).float().clone()
        source_labels = torch.nn.functional.interpolate(source_labels,
                                                 (source_feats.shape[2], source_feats.shape[3]), mode='nearest')
        source_labels = source_labels.squeeze(1).long()
        assert source_labels.shape[-1] == source_feats.shape[-1], '{} {}'.format(source_labels.shape, source_feats.shape)

        target_labels = target_labels.unsqueeze(1).float().clone()
        target_labels = torch.nn.functional.interpolate(target_labels,
                                                 (target_feats.shape[2], target_feats.shape[3]), mode='nearest')
        target_labels = target_labels.squeeze(1).long()
        assert target_labels.shape[-1] == target_feats.shape[-1], '{} {}'.format(target_labels.shape, target_feats.shape)

        # labels = labels.contiguous().view(batch_size, -1)
        # predict = predict.contiguous().view(batch_size, -1)
        source_feats = source_feats.permute(0, 2, 3, 1)
        source_feats = source_feats.contiguous().view(source_feats.shape[0], -1, source_feats.shape[-1])  # shape B*(H*W)*FeatDim
        # feats_, labels_ = self._new_hard_anchor_sampling(feats, labels, predict)
        source_feats_, source_labels_, source_selected_embedding_indices_list = self._dtm_anchor_sampling(source_feats, source_labels)

        target_feats = target_feats.permute(0, 2, 3, 1)
        target_feats = target_feats.contiguous().view(target_feats.shape[0], -1, target_feats.shape[-1])  # shape B*(H*W)*FeatDim
        # feats_, labels_ = self._new_hard_anchor_sampling(feats, labels, predict)
        target_feats_, target_labels_, target_selected_embedding_indices_list = self._dtm_target_anchor_sampling(target_feats, target_labels, boundary=self.configer['training']['target_boundary_sel'])

        source_loss = self._contrastive(source_feats_, source_labels_, queue=source_queue, iter=iter, domain="s", cl_lv=cl_lv)
        target_loss = self._contrastive(target_feats_, target_labels_, queue=source_queue, iter=iter, domain='t', cl_lv=cl_lv)

        if iter % self.configer['training']['val_interval'] == 0:
            self.visual.display_anchor_sampling_on_seg(source_labels.cpu().numpy(), source_selected_embedding_indices_list, "s", iter, cl_lv=cl_lv)
            self.visual.display_anchor_sampling_on_seg(target_labels.cpu().numpy(), target_selected_embedding_indices_list, "t", iter, cl_lv=cl_lv)

        # feats_ = torch.cat([source_feats_, target_feats_])
        # labels_ = torch.cat([source_labels_, target_labels_])
        # print("forward function _contrastive time cost:{}s".format(elapsed - _hard_anchor_sampling_elapsed))
        # print("----------------------------------")

        return source_loss, target_loss