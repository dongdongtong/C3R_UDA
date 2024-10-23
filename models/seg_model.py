import torch
import torch.nn as nn
import torch.nn.functional as F
import models.res_parts as mrp
import models.unet_parts as mup
import copy


from torch.nn import init
def init_weights(model, init_type='normal', gain=0.02, a=0.2):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=a, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)

    # propagate to children
    for m in model.children():
        if hasattr(m, 'init_weights'):
            m.init_weights(init_type, gain)


class SharedEncoder(nn.Module):
    def __init__(self, n_channels, n_classes, norm="InstanceNorm", skip=1, act="leaky_relu"):
        super(SharedEncoder, self).__init__()
        self.inc = mup.inconv(n_channels, 64, kernel_size=5, norm=norm, act=act)

        self.down1 = mrp.ResDown(64, 96, norm=norm, act=act)
        self.down2 = mrp.ResDown(96, 128, norm=norm, act=act)
        self.down3 = mrp.ResDown(128, 256, blocks=4, norm=norm, act=act)

        self.down3_ex = nn.Sequential(
            mrp.ResBlock(256, 512, 3, 1, dilation=1, norm=norm, act=act),
            mrp.ResBlock(512, 512, 3, 1, dilation=1, norm=norm, act=act),
            mrp.ResBlock(512, 512, 3, 1, dilation=2, norm=norm, act=act),
            mrp.ResBlock(512, 512, 3, 1, dilation=2, norm=norm, act=act),
            mup.double_conv(512, 512, 3, norm=norm, act=act)
        )

    def forward(self, input):
        encoder_feat_lv1 = self.inc(input)
        encoder_feat_lv2 = self.down1(encoder_feat_lv1)
        encoder_feat_lv3 = self.down2(encoder_feat_lv2)
        encoder_feat_lv4 = self.down3(encoder_feat_lv3)
        encoder_feat_lv4 = self.down3_ex(encoder_feat_lv4)
        
        return [encoder_feat_lv1, encoder_feat_lv2, encoder_feat_lv3, encoder_feat_lv4]


class SegDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, norm="InstanceNorm", skip=1, act="leaky_relu"):
        super(SegDecoder, self).__init__()

        self.up1 = mup.up(640, 128, norm=norm, act=act)  # output receptive field 186 * 186
        self.up2 = mup.up(224, 96, norm=norm, act=act)  # output receptive field 194 * 194
        self.up3 = mup.up(160, 64, norm=norm, act=act)  # output receptive field 198 * 198
 
        self.outc = mup.outconv(64, n_classes)

    def forward(self, multi_lv_features):
        encoder_feat_lv1, encoder_feat_lv2, encoder_feat_lv3, encoder_feat_lv4 = multi_lv_features

        segdec_feat_lv3 = self.up1(encoder_feat_lv4, encoder_feat_lv3)
        segdec_feat_lv2 = self.up2(segdec_feat_lv3, encoder_feat_lv2)
        segdec_feat_lv1 = self.up3(segdec_feat_lv2, encoder_feat_lv1)

        output = self.outc(segdec_feat_lv1)
        
        return output  # delete segdec_feat_lv1, no return midiet seg feats


class DilateResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, norm="InstanceNorm", skip=1, act="leaky_relu"):
        super(DilateResUNet, self).__init__()
        self.encoder = SharedEncoder(n_channels, n_classes, norm=norm, skip=1, act="leaky_relu")
        self.segdec = SegDecoder(n_channels, n_classes, norm=norm, skip=1, act="leaky_relu")
        self.aux_seg = Segmenter(in_channel=512, num_classes=n_classes)

    def forward(self, input, use_ds=True):
        multi_lv_features = self.encoder(input)
        output = self.segdec(multi_lv_features)
        aux_output = self.aux_seg(multi_lv_features[-1])

        if use_ds:
            return output, aux_output, multi_lv_features
        else:
            return output


class DilateResUNetCL(nn.Module):
    def __init__(self, n_channels, n_classes, norm="InstanceNorm", skip=1, act="leaky_relu", mem_size=5000, latent_dims=256):
        super(DilateResUNetCL, self).__init__()
        self.encoder = SharedEncoder(n_channels, n_classes, norm=norm, skip=1, act="leaky_relu")
        self.segdec = SegDecoder(n_channels, n_classes, norm=norm, skip=1, act="leaky_relu")
        self.aux_seg = Segmenter(in_channel=512, num_classes=n_classes)
        self.proj_head = ProjectionHead(512, latent_dims)

    def forward(self, input, use_ds=True):
        multi_lv_features = self.encoder(input)
        output = self.segdec(multi_lv_features)
        embeddings = self.proj_head(multi_lv_features[-1])
        aux_output = self.aux_seg(multi_lv_features[-1])

        return {"seg": output, "seg_aux": aux_output, "multi_lv_features": multi_lv_features, "embed": embeddings}


class DilateResUNetCLMem(nn.Module):
    def __init__(self, n_channels, n_classes, norm="InstanceNorm"):
        super(DilateResUNetCLMem, self).__init__()
        self.encoder = SharedEncoder(n_channels, n_classes, norm=norm, skip=1, act="leaky_relu")
        self.segdec = SegDecoder(n_channels, n_classes, norm=norm, skip=1, act="leaky_relu")
        self.aux_seg = Segmenter(in_channel=512, num_classes=n_classes)
        # self.proj_head = ProjectionHead(512, latent_dims)

    def forward(self, input, use_ds=True):
        multi_lv_features = self.encoder(input)
        output = self.segdec(multi_lv_features)  # only return hr level output
        # embeddings = self.proj_head(multi_lv_features[-1])
        aux_output = self.aux_seg(multi_lv_features[-1])

        return {
            "seg": output, 
            "seg_aux": aux_output, 
            "multi_lv_feats": multi_lv_features, 
            # "embeds": embeddings, 
            # "keys": embeddings.detach(),
        }


class DilateResUNetMultiLvCLMem(nn.Module):
    def __init__(self, n_channels, n_classes, norm="InstanceNorm", skip=1, act="leaky_relu", mem_size=5000, latent_dims=256):
        super(DilateResUNetMultiLvCLMem, self).__init__()
        self.encoder = SharedEncoder(n_channels, n_classes, norm=norm, skip=1, act=act)
        self.segdec = SegDecoderNoSeg(n_channels, n_classes, norm=norm, skip=1, act=act)

    def forward(self, input, use_ds=True):
        enc_multi_lv_features = self.encoder(input)
        dec_multi_lv_feats = self.segdec(enc_multi_lv_features)
        multi_lv_feats = [dec_multi_lv_feats['segdec_feat_hr'], dec_multi_lv_feats['segdec_feat_mr'], dec_multi_lv_feats['segdec_feat_mmr'], enc_multi_lv_features[-1]]

        return_out_dict = {
            "logits_hr": dec_multi_lv_feats['logits_hr'], 
            "logits_mr": dec_multi_lv_feats['logits_mr'],
            "logits_mmr": dec_multi_lv_feats['logits_mmr'],
            "logits_sr": dec_multi_lv_feats['logits_sr'],
            "multi_lv_features": multi_lv_feats,
        }

        return return_out_dict


class DilateResUNetCLMemMLPPH(nn.Module):
    def __init__(self, config):
        super(DilateResUNetCLMemMLPPH, self).__init__()
        self.config = config

        n_channels = self.config["size_C"]
        n_classes = self.config["n_class"]
        norm = self.config['seg_model']["normlization"]
        act = self.config['seg_model']["activation"]
        latent_dims = self.config['training']["proj_dim"]
        self.ignore_label = 5

        self.encoder = SharedEncoder(n_channels, n_classes, norm=norm, skip=1, act=act)
        self.segdec = SegDecoder(n_channels, n_classes, norm=norm, skip=1, act=act)
        self.aux_seg = Segmenter(in_channel=512, num_classes=n_classes)
        # self.proj_head = ProjectionHead(512, latent_dims)
        self.proj_head = MLPProjectionHead(512, latent_dims)

        # remove some old version code of pixel & seg mem buffer code

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size = y_hat.shape[0]

        X_ = None
        y_ = None

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_classes = torch.unique(this_y_hat)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y_hat == x).nonzero().shape[0] > 1]

            this_y = y[ii]
            for cls_i, cls_id in enumerate(this_classes):
                # if cls_id == 0 or cls_id == 2:
                #     continue
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                n_indices = num_hard + num_easy

                max_views = self.config['training']['max_views_bg'] if cls_id == 0 else self.config['training']['max_views']
                n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

                if num_hard >= n_sel_pixels_clas_id / 2 and num_easy >= n_sel_pixels_clas_id / 2:
                    num_hard_keep = n_sel_pixels_clas_id // 2
                    num_easy_keep = n_sel_pixels_clas_id - num_hard_keep
                elif num_hard >= n_sel_pixels_clas_id / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_sel_pixels_clas_id - num_easy_keep
                elif num_easy >= n_sel_pixels_clas_id / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_sel_pixels_clas_id - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_sel_pixels_clas_id))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]

                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                if type(X_) == type(None) and type(y_) == type(None):
                    X_ = X[ii, indices, :].squeeze()
                    y_ = torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)
                else:
                    X_ = torch.cat([X_, X[ii, indices, :].squeeze()], dim=0)
                    y_ = torch.cat([y_, torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)])

            return X_, y_
    
    def _random_anchor_sampling(self, feats, labels):
        batch_size = labels.shape[0]

        labels = labels.unsqueeze(1).float().clone()
        labels = F.interpolate(labels, size=(feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()

        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # shape B*(H*W)*FeatDim

        X_ = None
        y_ = None

        labels = labels.contiguous().view(batch_size, -1)

        for ii in range(batch_size):
            this_y_hat = labels[ii]
            this_classes = torch.unique(this_y_hat)
            this_classes = [x for x in this_classes if x != 5 and (this_y_hat == x).nonzero().shape[0] > 2]

            for cls_i, cls_id in enumerate(this_classes):
                # if cls_id == 0 or cls_id == 2:
                #     continue
                indices = (this_y_hat == cls_id).nonzero()

                n_indices = indices.shape[0]

                max_views = self.config['training']['max_views_bg'] if cls_id == 0 else self.config['training']['max_views']
                n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

                perm = torch.randperm(n_indices)
                selected_embedding_indices = indices[perm[:n_sel_pixels_clas_id]]

                # print()
                # print("anchor_sampling function feats[ii, selected_embedding_indices, :].squeeze() shape:", feats[ii, selected_embedding_indices, :].squeeze().shape)
                # print("-------------------------")

                if type(X_) == type(None) and type(y_) == type(None):
                    X_ = feats[ii, selected_embedding_indices, :].squeeze()
                    y_ = torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)
                else:
                    X_ = torch.cat([X_, feats[ii, selected_embedding_indices, :].squeeze()], dim=0)
                    y_ = torch.cat([y_, torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)])

        return X_, y_

    def _get_anchors(self, feats, labels, predicts):
        batch_size = labels.shape[0]

        labels = torch.argmax(labels, dim=-1)
        labels = labels.unsqueeze(1).float().clone()
        labels = F.interpolate(labels, size=(feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        
        labels_flatten = labels.contiguous().view(batch_size, -1)
        predicts_flatten = predicts.contiguous().view(batch_size, -1)

        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # shape B*(H*W)*FeatDim

        feats_, labels_ = self._hard_anchor_sampling(feats, labels_flatten, predicts_flatten)

        return feats_, labels_

    def forward(self, input, labels=None, hard_anchor_sampling=False, random_sampleing=False):
        multi_lv_features = self.encoder(input)
        output = self.segdec(multi_lv_features)
        aux_output = self.aux_seg(multi_lv_features[-1])

        out_dict = {
            "seg": output, 
            "seg_aux": aux_output, 
            "multi_lv_features": multi_lv_features, 
        }

        if type(labels) != type(None):
            sr_feats = multi_lv_features[-1]
            # proj_feats = self.proj_head(sr_feats)
            if hard_anchor_sampling:

                sr_output = F.interpolate(output, size=(sr_feats.shape[2], sr_feats.shape[3]), mode='bilinear')
                sr_predict = torch.argmax(torch.softmax(sr_output, 1), 1)
                pixel_feats_, y_ = self._get_anchors(sr_feats, labels, sr_predict)
                anchors_ = self.proj_head(pixel_feats_)
                out_dict['anchors_'] = anchors_
                out_dict['y_'] = y_
            if random_sampleing:
                pixel_feats_, y_ = self._random_anchor_sampling(sr_feats, labels)
                anchors_ = self.proj_head(pixel_feats_)
                out_dict['anchors_'] = anchors_
                out_dict['y_'] = y_

        return out_dict


class SegDecoderNoSeg(nn.Module):
    def __init__(self, n_channels, n_classes, norm="InstanceNorm", skip=1, act="leaky_relu"):
        super(SegDecoderNoSeg, self).__init__()

        self.up1 = mup.up(640, 128, norm=norm, act=act)  # output receptive field 186 * 186
        self.up2 = mup.up(224, 96, norm=norm, act=act)  # output receptive field 194 * 194
        self.up3 = mup.up(160, 64, norm=norm, act=act)  # output receptive field 198 * 198

        self.aux_seg_sr = Segmenter(in_channel=512, num_classes=n_classes)
        self.aux_seg_mmr = Segmenter(in_channel=128, num_classes=n_classes)
        self.aux_seg_mr = Segmenter(in_channel=96, num_classes=n_classes)
        self.seg_conv_hr = Segmenter(in_channel=64, num_classes=n_classes)

    def forward(self, multi_lv_features):
        encoder_feat_hr, encoder_feat_mr, encoder_feat_mmr, encoder_feat_sr = multi_lv_features

        segdec_feat_mmr = self.up1(encoder_feat_sr, encoder_feat_mmr)
        segdec_feat_mr = self.up2(segdec_feat_mmr, encoder_feat_mr)
        segdec_feat_hr = self.up3(segdec_feat_mr, encoder_feat_hr)

        logits_sr = self.aux_seg_sr(encoder_feat_sr)
        logits_mmr = self.aux_seg_mmr(segdec_feat_mmr)
        logits_mr = self.aux_seg_mr(segdec_feat_mr)
        logits_hr = self.seg_conv_hr(segdec_feat_hr)
        
        return {
            "logits_hr": logits_hr,
            "logits_mr": logits_mr,
            "logits_mmr": logits_mmr,
            "logits_sr": logits_sr,
            "segdec_feat_mmr": segdec_feat_mmr,
            "segdec_feat_mr": segdec_feat_mr,
            "segdec_feat_hr": segdec_feat_hr
        }


class MultiLvSeg(nn.Module):
    def __init__(self, in_channel=512, num_classes=5):
        super(MultiLvSeg, self).__init__()

        self.aux_seg_sr = Segmenter(in_channel=512, num_classes=num_classes)
        self.aux_seg_mmr = Segmenter(in_channel=128, num_classes=num_classes)
        self.aux_seg_mr = Segmenter(in_channel=96, num_classes=num_classes)
        self.seg_conv_hr = Segmenter(in_channel=64, num_classes=num_classes)

    def forward(self, multi_lv_seg_feats):  # remove upsample operation
        segdec_feat_hr, segdec_feat_mr, segdec_feat_mmr, segdec_feat_sr = multi_lv_seg_feats

        logits_sr = self.aux_seg_sr(segdec_feat_sr)
        logits_mmr = self.aux_seg_mmr(segdec_feat_mmr)
        logits_mr = self.aux_seg_mr(segdec_feat_mr)
        logits_hr = self.seg_conv_hr(segdec_feat_hr)

        return {
            "logits_hr": logits_hr,
            "logits_mr": logits_mr,
            "logits_mmr": logits_mmr,
            "logits_sr": logits_sr,
        }


class DilateResUNetMultiLvSeg(nn.Module):
    def __init__(self, config):
        super(DilateResUNetMultiLvSeg, self).__init__()
        self.config = config

        n_channels=self.config["size_C"]
        n_classes=self.config["n_class"]
        norm=self.config['seg_model']["normlization"]
        act=self.config['seg_model']["activation"]
        self.ignore_label = 5

        self.encoder = SharedEncoder(n_channels, n_classes, norm=norm, skip=1, act=act)
        self.segdec = SegDecoderNoSeg(n_channels, n_classes, norm=norm, skip=1, act=act)
        self.proj_head = MLPProjectionHead(64 + 96 + 128 + 512, config['training']['proj_dim'])
        # self.proj_head = MultiProjectionHead(self.config['seg_model']['enc_multi_lv_dims'])
    
    def _get_scaled_location(self, index, cur_H, cur_W, ds_stride):
        # scaled_loc_x = (cur_loc // cur_H)
        # scaled_loc_y = cur_loc - scaled_loc_x * cur_H
        cur_index_x = index // cur_W
        cur_index_y = index - cur_index_x * cur_W

        scaled_index_x = cur_index_x // ds_stride
        scaled_index_y = cur_index_y // ds_stride
        ds_H = cur_H // ds_stride
        ds_W = cur_W // ds_stride

        scaled_index = (scaled_index_x * ds_W) + scaled_index_y

        return scaled_index

    def _get_all_res_cat_anchors(self, indices, multi_lv_feat):
        # recurve to select multi lv pixels from corresponding feature maps
        cur_lv_indices = indices
        selected_pixels_cat = None
        for i, this_lv_feat in enumerate(multi_lv_feat):
            this_lv_feat_permute = this_lv_feat.reshape(this_lv_feat.shape[0], -1).transpose(0,1)

            if type(selected_pixels_cat) == type(None):
                selected_pixels_cat = this_lv_feat_permute[cur_lv_indices].squeeze(1)
            else:
                selected_pixels_cat = torch.cat([selected_pixels_cat, this_lv_feat_permute[cur_lv_indices].squeeze(1)], dim=1)
            
            cur_lv_indices = self._get_scaled_location(cur_lv_indices, this_lv_feat.shape[1], this_lv_feat.shape[2], 2)

        return selected_pixels_cat

    def _hard_anchor_sampling(self, multi_lv_feats, labels, predicts):
        batch_size = labels.shape[0]

        X_ = None
        y_ = None

        for ii in range(batch_size):
            this_label = labels[ii]
            this_classes = torch.unique(this_label)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_label == x).nonzero().shape[0] > 1]

            this_pred = predicts[ii]
            for cls_i, cls_id in enumerate(this_classes):
                # if cls_id == 0 or cls_id == 2:
                #     continue
                all_indices = (this_label == cls_id).nonzero()
                hard_indices = ((this_label == cls_id) & (this_pred != cls_id)).nonzero()
                easy_indices = ((this_label == cls_id) & (this_pred == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                n_indices = num_hard + num_easy

                max_views = self.config['training']['max_views_bg'] if cls_id == 0 else self.config['training']['max_views']
                n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

                if num_hard >= n_sel_pixels_clas_id / 2 and num_easy >= n_sel_pixels_clas_id / 2:
                    num_hard_keep = n_sel_pixels_clas_id // 2
                    num_easy_keep = n_sel_pixels_clas_id - num_hard_keep
                elif num_hard >= n_sel_pixels_clas_id / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_sel_pixels_clas_id - num_easy_keep
                elif num_easy >= n_sel_pixels_clas_id / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_sel_pixels_clas_id - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_sel_pixels_clas_id))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]

                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                this_multi_lv_feat = [this_lv_feats[ii] for this_lv_feats in multi_lv_feats]
                if type(X_) == type(None) and type(y_) == type(None):
                    X_ = self._get_all_res_cat_anchors(indices, this_multi_lv_feat)
                    y_ = torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)
                else:
                    X_ = torch.cat([X_, self._get_all_res_cat_anchors(indices, this_multi_lv_feat)], dim=0)
                    y_ = torch.cat([y_, torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)])

            return X_, y_
    
    def _random_anchor_sampling(self, multi_lv_feats, labels):
        batch_size = labels.shape[0]

        X_ = None
        y_ = None

        labels = labels.contiguous().view(batch_size, -1)

        for ii in range(batch_size):
            this_label = labels[ii]
            this_classes = torch.unique(this_label)
            this_classes = [x for x in this_classes if x != 5]
            this_classes = [x for x in this_classes if (this_label == x).nonzero().shape[0] > 2]

            for cls_i, cls_id in enumerate(this_classes):
                indices = (this_label == cls_id).nonzero()
                n_indices = indices.shape[0]

                max_views = self.config['training']['max_views_bg'] if cls_id == 0 else self.config['training']['max_views']
                n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

                perm = torch.randperm(n_indices)
                selected_embedding_indices = indices[perm[:n_sel_pixels_clas_id]]

                this_multi_lv_feat = [this_lv_feats[ii] for this_lv_feats in multi_lv_feats]
                if type(X_) == type(None) and type(y_) == type(None):
                    X_ = self._get_all_res_cat_anchors(selected_embedding_indices, this_multi_lv_feat)
                    y_ = torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)
                else:
                    X_ = torch.cat([X_, self._get_all_res_cat_anchors(selected_embedding_indices, this_multi_lv_feat)], dim=0)
                    y_ = torch.cat([y_, torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)])

        return X_, y_

    def _get_multi_lv_cat_feats(self, multi_lv_feats, labels, hr_predicts):
        batch_size = labels.shape[0]
        labels = torch.argmax(labels, dim=-1)
        labels = labels.contiguous().view(batch_size, -1)
        hr_predicts = hr_predicts.contiguous().view(batch_size, -1)

        feats_, labels_ = self._hard_anchor_sampling(multi_lv_feats, labels, hr_predicts)

        return feats_, labels_

    def forward(self, input, labels=None, hard_anchor_sampling=False, random_sampleing=False):
        enc_multi_lv_features = self.encoder(input)
        multi_lv_logits = self.segdec(enc_multi_lv_features)
        # embeddings = self.proj_head(multi_lv_features[-1])
        dec_multi_lv_feats = [multi_lv_logits['segdec_feat_hr'], multi_lv_logits['segdec_feat_mr'], multi_lv_logits['segdec_feat_mmr'], enc_multi_lv_features[-1]]

        # multi_lv_logits = self.multi_lv_seg(dec_multi_lv_feats)

        return_out_dict = {
            "logits_hr": multi_lv_logits['logits_hr'], 
            "logits_mr": multi_lv_logits['logits_mr'],
            "logits_mmr": multi_lv_logits['logits_mmr'],
            "logits_sr": multi_lv_logits['logits_sr'],
            "multi_lv_features": dec_multi_lv_feats
            }

        if type(labels) != type(None):
            if hard_anchor_sampling:
                hr_predict = torch.argmax(torch.softmax(multi_lv_logits['logits_hr'], 1), 1)
                multi_lv_cat_feats_, multi_lv_cat_feats_labels_ = self._get_multi_lv_cat_feats(enc_multi_lv_features, labels, hr_predict)
                multi_lv_cat_proj_keys = self.proj_head(multi_lv_cat_feats_)
            elif random_sampleing:
                multi_lv_cat_feats_, multi_lv_cat_feats_labels_ = self._random_anchor_sampling(enc_multi_lv_features, labels)
                multi_lv_cat_proj_keys = self.proj_head(multi_lv_cat_feats_)
            
            return_out_dict['multi_lv_cat_proj_keys'] = multi_lv_cat_proj_keys
            return_out_dict['multi_lv_cat_feats_labels_'] = multi_lv_cat_feats_labels_

        return return_out_dict


class MultiProjectionHead(nn.Module):
    def __init__(self, enc_multi_lv_dims, latent_dims):
        super(MultiProjectionHead, self).__init__()
        self.multi_lv_proj_heads = nn.ModuleList()
        for this_lv_dims in enc_multi_lv_dims:
            self.multi_lv_proj_heads.append(ProjectionHead(this_lv_dims, min(this_lv_dims, latent_dims)))
    
    def forward(self, multi_lv_feats):
        multi_lv_feats_proj = []
        for this_proj_head, this_lv_feat in zip(self.multi_lv_proj_heads, multi_lv_feats):
            multi_lv_feats_proj.append(this_proj_head(this_lv_feat))
        
        return multi_lv_feats_proj


class EMA(object):  # Momentum Teacher Network
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        if self.step == 0:
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.copy_(param.data)  # initialize
                ema_param.requires_grad = False  # not update by gradient
            self.step += 1
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1


class DilateResUNetNoDs(nn.Module):
    def __init__(self, n_channels, n_classes, norm="InstanceNorm", skip=1, act="leaky_relu"):
        super(DilateResUNetNoDs, self).__init__()
        self.encoder = SharedEncoder(n_channels, n_classes, norm=norm, skip=1, act="leaky_relu")
        self.segdec = SegDecoder(n_channels, n_classes, norm=norm, skip=1, act="leaky_relu")
        self.aux_seg = Segmenter(in_channel=512, num_classes=n_classes)

    def forward(self, input, use_ds=True):
        multi_lv_features = self.encoder(input)
        output = self.segdec(multi_lv_features)
        aux_output = self.aux_seg(multi_lv_features[-1])

        if use_ds:
            return output, aux_output, multi_lv_features
        else:
            return output


class ResUNetDecoder(nn.Module):
    def __init__(self, n_classes, norm="InstanceNorm", skip=1, act="leaky_relu"):
        super(ResUNetDecoder, self).__init__()
        self.skip = skip

        if norm == "BatchNorm":
            self.use_bias = False
        elif norm == "InstanceNorm":
            self.use_bias = True

        self.up_act = mrp.ResUp(512, 256, norm=norm, up=False, act=act)

        self.up1 = mrp.ResUp(256, 256, norm=norm, act=act)     # 64 * 64
        self.up2 = mrp.ResUp(384, 128, norm=norm, act=act)     # 128 * 128
        self.up3 = mrp.ResUp(224, 96, norm=norm, act=act)        # 256 * 256

        self.up4 = mrp.ResUp(160, 64, norm=norm, up=False, act=act)      # 256 * 256

        self.conv_act = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=self.use_bias),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True) if act == "leaky_relu" else nn.ReLU(True)
        )

        self.outc = nn.Conv2d(32, n_classes, 3, 1, 1)

    def forward(self, multi_lv_features):
        encoder_feat_lv1, encoder_feat_lv2, encoder_feat_lv3, encoder_feat_lv4 = multi_lv_features

        imgdec_feat_lv4 = self.up_act(encoder_feat_lv4)  # act conv
        imgdec_feat_lv3 = torch.cat([self.up1(imgdec_feat_lv4), encoder_feat_lv3], dim=1)
        imgdec_feat_lv2 = torch.cat([self.up2(imgdec_feat_lv3), encoder_feat_lv2], dim=1)
        imgdec_feat_lv1 = torch.cat([self.up3(imgdec_feat_lv2), encoder_feat_lv1], dim=1)

        # output conv
        output = self.up4(imgdec_feat_lv1)
        output = self.conv_act(output)
        output = self.outc(output)
        # if cl_encode:
        #     return x7, torch.sigmoid(output)

        return torch.tanh(output)


class ResUNetDecoderNoSkipC(nn.Module):
    def __init__(self, n_classes, norm="InstanceNorm", skip=1, act="leaky_relu"):
        super(ResUNetDecoderNoSkipC, self).__init__()
        self.skip = skip

        if norm == "BatchNorm":
            self.use_bias = False
        elif norm == "InstanceNorm":
            self.use_bias = True

        self.up_act = mrp.ResUp(512, 256, norm=norm, up=False, act=act)

        self.up1 = mrp.ResUp(256, 256, norm=norm, act=act)     # 64 * 64
        self.up2 = mrp.ResUp(256, 128, norm=norm, act=act)     # 128 * 128
        self.up3 = mrp.ResUp(128, 96, norm=norm, act=act)        # 256 * 256

        self.up4 = mrp.ResUp(96, 64, norm=norm, up=False, act=act)      # 256 * 256

        self.conv_act = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=self.use_bias),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True) if act == "leaky_relu" else nn.ReLU(True)
        )

        self.outc = nn.Conv2d(32, n_classes, 3, 1, 1)

    def forward(self, multi_lv_features):
        _, _, _, encoder_feat_lv4 = multi_lv_features

        imgdec_feat_lv4 = self.up_act(encoder_feat_lv4)  # act conv
        imgdec_feat_lv3 = self.up1(imgdec_feat_lv4)
        imgdec_feat_lv2 = self.up2(imgdec_feat_lv3)
        imgdec_feat_lv1 = self.up3(imgdec_feat_lv2)

        # output conv
        output = self.up4(imgdec_feat_lv1)
        output = self.conv_act(output)
        output = self.outc(output)
        # if cl_encode:
        #     return x7, torch.sigmoid(output)

        return torch.tanh(output)


class Segmenter(nn.Module):
    def __init__(self, in_channel=512, num_classes=5):
        super(Segmenter, self).__init__()

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channel, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, inputs):  # remove upsample operation
        out = self.out_conv(inputs)

        return out

# add ProjectionHead for project the features to l2 normed embedding space
class ProjectionHead(nn.Module):
    def __init__(self, dim_in=256, project_dims=256):
        super(ProjectionHead, self).__init__()

        self.project_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            nn.InstanceNorm2d(dim_in),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(dim_in, project_dims, kernel_size=1),
        )

    def forward(self, inputs):
        embddings = self.project_conv(inputs)
        return F.normalize(embddings, p=2, dim=1)


# add ProjectionHead for project the features to l2 normed embedding space
class MLPProjectionHead(nn.Module):
    def __init__(self, dim_in=256, project_dims=256):
        super(MLPProjectionHead, self).__init__()

        self.project_mlp = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(0.2, True),
            nn.Linear(dim_in, project_dims),
            nn.LeakyReLU(0.2, True),
            nn.Linear(project_dims, project_dims),
        )

    def forward(self, inputs):
        embddings = self.project_mlp(inputs)
        return F.normalize(embddings, p=2, dim=1)