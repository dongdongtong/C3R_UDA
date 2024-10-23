import itertools
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from losses.contrastive_loss_multi_lv_feat_cat_proj import MultiLvPixelContrastLoss, MultiLvPixelContrastLossImgMem, MultiLvPixelContrastLossMem

from models.discriminator import Discriminator, DomainClassifier
from models.seg_model import EMA, DilateResUNet, DilateResUNetCLMem, DilateResUNetMultiLvSeg, MultiProjectionHead, ResUNetDecoder
from utils.utils import LambdaLROnStep
from losses import GANLoss, TaskLoss
from utils.utils import one_hot

import tsnecuda
import matplotlib.pyplot as plt
import os


class TrainModel():
    def __init__(self, config, writer, logger, visual, logdir):
        self.config = config
        self.writer = writer
        self.logger = logger
        self.visual = visual
        self.logdir = logdir

        self.nets = []
        self.nets_DP = []
        self.optimizers = []
        self.lr_schedulers = []

        self.init_nets()
        self.init_optimizers()
        self.ema = EMA(self.seg_net_DP, self.config['training']['ema_net_momentum'])

        # losses
        self.task_loss = TaskLoss(num_classes=self.config['n_class'])
        self.gan_loss = GANLoss()
        self.cycle_loss = torch.nn.L1Loss()
        self.identity_loss = torch.nn.L1Loss()
        self.contrast_loss_imgmem = MultiLvPixelContrastLossImgMem(self.config, self.writer)
        self.contrast_loss = MultiLvPixelContrastLoss(self.config, self.writer)
        self.contrast_loss_pixelmem = MultiLvPixelContrastLossMem(self.config, self.writer)

        # fake images pool
        self.num_fakes = 0
        self.fake_T_images_pool = torch.zeros(
            (config['training']["pool_size"], 
            config['data']['source']["batch_size"], 
            config["size_C"], config["size_H"], config["size_W"]
            )
        )
        self.fake_S_images_pool = torch.zeros(
            (config['training']["pool_size"], 
            config['data']['target']["batch_size"], 
            config["size_C"], config["size_H"], config["size_W"]
            )
        )

        # add the memory buffer for paried-pixel contrastive learning
        self.segment_queue = torch.randn(config['n_class'], config['training']['segment_memory_size'], config['training']['proj_dim'], requires_grad=False)
        self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2).cuda()
        self.segment_queue_ptr = torch.zeros(config['n_class'], dtype=torch.long, requires_grad=False)

        self.pixel_queue = torch.randn(config['n_class'], config['training']['pixel_memory_size'], config['training']['proj_dim'], requires_grad=False)
        self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2).cuda()
        self.pixel_queue_ptr = torch.zeros(config['n_class'], dtype=torch.long, requires_grad=False)

        self.source_images_pool_ptr = torch.zeros(config['n_class'], dtype=torch.long)
        self.source_images_pool = torch.zeros(
            (config['n_class'], config['training']['image_memory_size'], 
            config["size_C"], config["size_H"], config["size_W"])
        )
        self.source_labels_pool = torch.zeros(
            (config['n_class'], config['training']['image_memory_size'], 
            config["size_H"], config["size_W"])
        )

        # add the t-sne memory bank for storing the pixels before classifiers
        # self.latent_pixel_queue = torch.randn(config['n_class'], self.config['training']['tsne_memory_size'], 512, requires_grad=False).cuda()
        # self.latent_pixel_queue_ptr = torch.zeros(config['n_class'], dtype=torch.long, requires_grad=False)

        # self.out_pixel_queue = torch.randn(config['n_class'], self.config['training']['tsne_memory_size'], 64, requires_grad=False).cuda()
        # self.out_pixel_queue_ptr = torch.zeros(config['n_class'], dtype=torch.long, requires_grad=False)


        # training process param
        self.best_target_dice = 0.
        self.iter = 0

        # contrast process param
        self.with_contrast = config['training']["contrast"]
        self.pixel_memory_size = config['training']["pixel_memory_size"]
        self.segment_memory_size = config['training']["segment_memory_size"]
        self.pixel_update_freq_bg = config['training']["pixel_update_freq_bg"]  # backgroud low update frequency
        self.pixel_update_freq_fg = config['training']["pixel_update_freq_fg"]  # foregroud high update frequency
        self.contrast_warmup_iters = config['training']["contrast_start_step"]

        self.network_stride = 8

        # self.logger.info("with_contrast: {}, warmup_iters: {}, with_memory_size: {}".format(
        #     self.with_contrast, self.contrast_warmup_iters, self.memory_size))
    
    def set_input(self, source_images, source_labels, target_images, target_labels):
        self.source_images = source_images
        self.source_labels = source_labels
        self.target_images = target_images
        self.target_labels = target_labels
    
    def seg_model_backward(self):
        loss_G = torch.Tensor([0]).cuda()
        log_losses = dict()

        self.source_out_dict = self.seg_forward(self.source_images, torch.argmax(self.source_labels, dim=-1), hard_anchor_sampling=False, random_sampleing=True)
        self.target_out_dict = self.seg_forward(self.target_images)

        # compute source seg loss
        loss_source_seg_ce, loss_source_seg_dc = self.task_loss(self.source_out_dict['logits_hr'], self.source_labels)
        loss_G += self.config['training']['lambda_seg']['hr'] * (loss_source_seg_ce + loss_source_seg_dc)

        log_losses["seg/loss_source_seg_ce"] = loss_source_seg_ce.detach()
        log_losses["seg/loss_source_seg_dc"] = loss_source_seg_dc.detach()

        # compute source aux seg loss
        if self.config['training']['lambda_seg']['aux']:
            self.source_seg_mr_up = F.interpolate(self.source_out_dict['logits_mr'], (256, 256), mode="bilinear")
            self.source_seg_mmr_up = F.interpolate(self.source_out_dict['logits_mmr'], (256, 256), mode="bilinear")
            self.source_seg_sr_up = F.interpolate(self.source_out_dict['logits_sr'], (256, 256), mode="bilinear")

            loss_source_seg_mr_ce, loss_source_seg_mr_dc = self.task_loss(self.source_seg_mr_up, self.source_labels)
            loss_source_seg_mmr_ce, loss_source_seg_mmr_dc = self.task_loss(self.source_seg_mmr_up, self.source_labels)
            loss_source_seg_sr_ce, loss_source_seg_sr_dc = self.task_loss(self.source_seg_sr_up, self.source_labels)

            loss_G += (self.config['training']['lambda_seg']['mr'] * (loss_source_seg_mr_ce + loss_source_seg_mr_dc) +
                       self.config['training']['lambda_seg']['mmr'] * (loss_source_seg_mmr_ce + loss_source_seg_mmr_dc) +
                       self.config['training']['lambda_seg']['sr'] * (loss_source_seg_sr_ce + loss_source_seg_sr_dc))
        
        # compute pred adv loss
        if self.iter > self.config['training']['start_adv_p']:
            if self.config['training']['lambda_adv_p'] > 0:
                loss_target_seg_adv = self.gan_loss(self.dis_pred_forward(self.target_out_dict['seg']), True)
                loss_G += self.config['training']['lambda_adv_p'] * loss_target_seg_adv

                log_losses["seg_adv/loss_target_seg_adv"] = loss_target_seg_adv.detach()
            
            if self.config['training']['lambda_adv_aux_p'] > 0 and self.config['training']['lambda_aux_seg'] > 0:
                loss_target_aux_seg_adv = self.gan_loss(self.dis_aux_pred_forward(self.target_out_dict['seg_aux']), True)
                loss_G += self.config['training']['lambda_adv_aux_p'] * loss_target_aux_seg_adv

                log_losses["seg_adv/loss_target_aux_seg_adv"] = loss_target_aux_seg_adv.detach()
        
        # bidirectional translation
        if self.config['training']['lambda_s2t_seg'] > 0:
            self.fake_T = self.dec_img_toT_forward(self.source_out_dict['multi_lv_feats'])
            self.fake_S = self.dec_img_toS_forward(self.target_out_dict['multi_lv_feats'])

            # compute img adv loss
            loss_fake_T_img_adv = self.gan_loss(self.dis_img_T_forward(self.fake_T), True)
            loss_fake_S_img_adv = self.gan_loss(self.dis_img_S_forward(self.fake_S), True)
            loss_G += (self.config['training']['lambda_adv_image_s'] * loss_fake_T_img_adv +\
                self.config['training']['lambda_adv_image_t'] * loss_fake_S_img_adv)

            log_losses["img_adv/loss_fake_T_img_adv"] = loss_fake_T_img_adv.detach()
            log_losses["img_adv/loss_fake_S_img_adv"] = loss_fake_S_img_adv.detach()

            self.fake_T_out_dict = self.seg_forward(self.fake_T)
            self.fake_S_out_dict = self.seg_forward(self.fake_S)

            # compute source seg loss
            loss_fake_T_seg_ce, loss_fake_T_seg_dc = self.task_loss(self.fake_T_out_dict['seg'], self.source_labels)
            loss_G += self.config['training']['lambda_s2t_seg'] * (loss_fake_T_seg_ce + loss_fake_T_seg_dc)

            log_losses["seg/loss_fake_T_seg_ce"] = loss_fake_T_seg_ce.detach()
            log_losses["seg/loss_fake_T_seg_dc"] = loss_fake_T_seg_dc.detach()

            # compute source aux seg loss
            if self.config['training']['lambda_aux_seg'] > 0:
                loss_fake_T_aux_seg_ce, loss_fake_T_aux_seg_dc = self.task_loss(self.fake_T_out_dict['seg_aux'], self.source_labels)
                loss_G += self.config['training']['lambda_aux_seg'] * (loss_fake_T_aux_seg_ce + loss_fake_T_aux_seg_dc)
            
            # keep consistent: fake_S pred and target pred
            if self.config['training']["lambda_consis_pred"] > 0 and self.iter > self.config['training']["consis_pred_start_step"]:
                self.target_pseudo_label = self.get_pseodu_labels(self.target_out_dict['seg'])

                loss_seg_consis_ce, loss_seg_consis_dc = self.task_loss(self.fake_S_out_dict['seg'], self.target_pseudo_label)
                loss_G += self.config['training']["lambda_consis_pred"] * (loss_seg_consis_ce + loss_seg_consis_dc)

                if self.config['training']['lambda_aux_seg']:
                    loss_aux_seg_consis_ce, loss_aux_seg_consis_dc = self.task_loss(self.fake_S_out_dict['seg_aux'], self.target_pseudo_label)
                    loss_G += self.config['training']["lambda_consis_pred"] * self.config['training']['lambda_aux_seg'] * (loss_aux_seg_consis_ce + loss_aux_seg_consis_dc)

                log_losses["consistency/loss_seg_consis_ce"] = loss_seg_consis_ce.detach()
                log_losses["consistency/loss_seg_consis_dc"] = loss_seg_consis_dc.detach()
            
            # compute fake_S pred adv loss
            if self.iter > self.config['training']['start_adv_p']:
                if self.config['training']['lambda_adv_tp'] > 0:
                    loss_fake_S_seg_adv = self.gan_loss(self.dis_pred_forward(self.fake_S_out_dict['seg']), True)
                    loss_G += self.config['training']['lambda_adv_tp'] * loss_fake_S_seg_adv

                    log_losses["seg_adv/loss_fake_S_seg_adv"] = loss_fake_S_seg_adv.detach()
                
                if self.config['training']['lambda_adv_aux_tp'] > 0 and self.config['training']['lambda_aux_seg'] > 0:
                    loss_fake_S_aux_seg_adv = self.gan_loss(self.dis_aux_pred_forward(self.fake_S_out_dict['seg_aux']), True)
                    loss_G += self.config['training']['lambda_adv_aux_tp'] * loss_fake_S_aux_seg_adv

                    log_losses["seg_adv/loss_fake_S_aux_seg_adv"] = loss_fake_S_aux_seg_adv.detach()
            
            # compute img cycle loss
            if self.config['training']["lambda_cycle_t"] > 0 and self.config['training']["lambda_cycle_s"] > 0:
                self.cyc_S = self.dec_img_toS_forward(self.fake_T_out_dict['multi_lv_feats'])
                self.cyc_T = self.dec_img_toT_forward(self.fake_S_out_dict['multi_lv_feats'])
                
                loss_cyc_S = self.cycle_loss(self.cyc_S, self.source_images)
                loss_cyc_T = self.cycle_loss(self.cyc_T, self.target_images)
                loss_G += (self.config['training']["lambda_cycle_s"] * loss_cyc_S + self.config['training']["lambda_cycle_t"] * loss_cyc_T)

                log_losses["cyc/loss_cyc_S"] = loss_cyc_S.detach()
                log_losses["cyc/loss_cyc_T"] = loss_cyc_T.detach()
            
            # compute img identity loss
            if self.config['training']["lambda_id_t"] > 0 and self.config['training']["lambda_id_s"] > 0:
                self.id_S = self.dec_img_toS_forward(self.source_out_dict['multi_lv_feats'])
                self.id_T = self.dec_img_toT_forward(self.target_out_dict['multi_lv_feats'])
                
                loss_id_S = self.identity_loss(self.id_S, self.source_images)
                loss_id_T = self.identity_loss(self.id_T, self.target_images)
                loss_G += (self.config['training']["lambda_id_s"] * loss_id_S + self.config['training']["lambda_id_t"] * loss_id_T)

                log_losses["id/loss_id_S"] = loss_id_S.detach()
                log_losses["id/loss_id_T"] = loss_id_T.detach()

        # pixel-level contrast
        if self.config['training']['contrast'] and self.iter > self.config['training']['contrast_start_step']:
            # mem queue update
            if self.config['training']['is_image_mem']:
                self._image_in_pool(self.source_images, self.source_labels)
            else:
                with torch.no_grad():
                    self.ema.update(self.seg_net_DP)
                    ema_out = self.ema.model(self.source_images)
                    self._dequeue_and_enqueue(ema_out['keys'], torch.argmax(self.source_labels, -1))

            # predict = torch.argmax(torch.softmax(self.source_out_dict['seg'], 1), 1)
            multi_lv_embeddings = self.source_out_dict['multi_lv_cat_proj_keys']
            multi_lv_embeddings_labels = self.source_out_dict['multi_lv_cat_feats_labels_']

            if self.iter > self.config['training']['mem_start_step']:
                if self.config['training']['is_image_mem']:
                    queue_, queue_y_ = self.get_emb_queue_from_imgmem()
                    # print("use img mem")
                    loss_source_contrast = self.contrast_loss_imgmem(multi_lv_embeddings, multi_lv_embeddings_labels, queue_, queue_y_, self.iter)
                else:
                    loss_source_contrast = self.contrast_loss_pixelmem(multi_lv_embeddings, torch.argmax(self.source_labels, -1), predict, self.pixel_queue, self.iter)
            else:
                loss_source_contrast = self.contrast_loss(multi_lv_embeddings, multi_lv_embeddings_labels, self.iter)
            
            loss_G += self.config['training']['lambda_contrast'] * loss_source_contrast

            log_losses["contrast/source_contrast_loss"] = loss_source_contrast.detach()
            with torch.no_grad():
                self.writer.add_scalar("contrast/std_value_on_each_channel", 
                    torch.std(multi_lv_embeddings.reshape(self.config['training']['proj_dim'], -1), dim=1).abs().mean(), self.iter)

        if loss_G.item() != 0:
            loss_G.backward(retain_graph=(not self.config['training']['lambda_detach_cyc']))  # if w/o detached cyc loss, then retain graph
            self.visual.plot_current_errors(log_losses, self.iter)

            if self.config['training']['histogram'] and self.iter % self.config['training']['val_interval'] == 0 or \
                    (self.iter + 1) == self.config['training']['train_iters']:
                for name, params in self.seg_net.named_parameters():
                    self.writer.add_histogram("weight/"+name, params, self.iter)
                    # print("add grad histogram name:", name, "grad type:", type(params.grad))
                    self.writer.add_histogram("grad/"+name, params.grad, self.iter)

    def dec_backward(self):
        loss_dec = torch.Tensor([0]).cuda()
        log_losses = dict()

        if self.config['training']["lambda_id_t"] > 0 and self.config['training']["lambda_id_s"] > 0:
            id_S = self.dec_img_toS_forward(self.detach(self.source_out_dict['multi_lv_feats']))
            id_T = self.dec_img_toT_forward(self.detach(self.target_out_dict['multi_lv_feats']))

            loss_id_S = self.identity_loss(id_S, self.source_images)
            loss_id_T = self.identity_loss(id_T, self.target_images)
            loss_dec += self.config['training']["lambda_id_t"] * loss_id_T + self.config['training']["lambda_id_s"] * loss_id_S

            log_losses["id/dec_id_loss_T"] = loss_id_T.detach()
            log_losses["id/dec_id_loss_S"] = loss_id_S.detach()
        
        if self.config['training']["lambda_cycle_t"] > 0 and self.config['training']["lambda_cycle_s"] > 0:
            cyc_S = self.dec_img_toS_forward(self.detach(self.fake_T_out_dict['multi_lv_feats']))
            cyc_T = self.dec_img_toT_forward(self.detach(self.fake_S_out_dict['multi_lv_feats']))

            loss_cyc_S = self.cycle_loss(cyc_S, self.source_images)
            loss_cyc_T = self.cycle_loss(cyc_T, self.target_images)
            loss_dec += (self.config['training']["lambda_cycle_s"] * loss_cyc_S + self.config['training']["lambda_cycle_t"] * loss_cyc_T)

            log_losses["cyc/decoder_cyc_loss_T"] = loss_cyc_T.detach()
            log_losses["cyc/decoder_cyc_loss_S"] = loss_cyc_S.detach()
        
        fake_T = self.dec_img_toT_forward(self.detach(self.source_multi_lv_feats))
        fake_S = self.dec_img_toS_forward(self.detach(self.target_multi_lv_feats))

        loss_fake_T_img_adv = self.gan_loss(self.dis_img_T_forward(fake_T), True)
        loss_fake_S_img_adv = self.gan_loss(self.dis_img_S_forward(fake_S), True)
        loss_gan_adv = self.config['training']['lambda_adv_image_s'] * loss_fake_T_img_adv +\
             self.config['training']['lambda_adv_image_t'] * loss_fake_S_img_adv
        # loss_gan_adv = self.gan_loss(self.dis_img_S_forward(fake_S), True) + self.gan_loss(self.dis_img_T_forward(fake_T), True)
        loss_dec += loss_gan_adv

        if loss_dec.item() != 0:
            loss_dec.backward()
        self.visual.plot_current_errors(log_losses, self.iter)

        self.fake_T_from_pool = self.fake_images_pool(self.num_fakes, self.fake_T.data.cpu(), self.fake_T_images_pool).cuda()
        self.fake_S_from_pool = self.fake_images_pool(self.num_fakes, self.fake_S.data.cpu(), self.fake_S_images_pool).cuda()
        self.num_fakes += 1

    def img_dis_backward(self):
        loss_img_dis = torch.Tensor([0]).cuda()
        log_losses = dict()

        fake_T_is_fake = self.dis_img_T_forward(self.fake_T_from_pool)
        real_T_is_real = self.dis_img_T_forward(self.target_images)

        fake_S_is_fake = self.dis_img_S_forward(self.fake_S_from_pool)
        real_S_is_real = self.dis_img_S_forward(self.source_images)

        loss_img_dis_T = self.gan_loss(fake_T_is_fake, False) + self.gan_loss(real_T_is_real, True)
        loss_img_dis_S = self.gan_loss(fake_S_is_fake, False) + self.gan_loss(real_S_is_real, True)
        loss_img_dis = loss_img_dis_T + loss_img_dis_S

        log_losses["dis/loss_img_dis_S"] = loss_img_dis_S.detach()
        log_losses["dis/loss_img_dis_T"] = loss_img_dis_T.detach()
        
        if loss_img_dis.item() != 0:
            loss_img_dis.backward()
        self.visual.plot_current_errors(log_losses, self.iter)

    def pred_dis_backward(self):
        loss_pred_dis = torch.Tensor([0]).cuda()
        log_losses = dict()

        if self.config['training']['lambda_adv_p'] > 0:
            source_seg_is_real = self.dis_pred_forward(self.source_logits.detach())
            target_seg_is_fake = self.dis_pred_forward(self.target_logits.detach())

            loss_pred_dis = 0.5 * (self.gan_loss(target_seg_is_fake, False) + self.gan_loss(source_seg_is_real, True))

            log_losses["dis/loss_aux_pred_dis"] = loss_pred_dis.detach()

            if self.config['training']['lambda_adv_aux_p'] > 0 and self.config['training']['lambda_aux_seg'] > 0:
                source_aux_seg_is_real = self.dis_aux_pred_forward(self.source_aux_logits.detach())
                target_aux_seg_is_fake = self.dis_aux_pred_forward(self.target_aux_logits.detach())

                loss_aux_pred_dis = 0.5 * (self.gan_loss(target_aux_seg_is_fake, False) + self.gan_loss(source_aux_seg_is_real, True))
                loss_pred_dis += loss_aux_pred_dis

                log_losses["dis/loss_aux_pred_dis"] = loss_aux_pred_dis.detach()

        if self.config['training']['lambda_adv_tp'] > 0:
            fake_T_seg_is_real = self.dis_pred_forward(self.fake_T_logits.detach())
            fake_S_seg_is_fake = self.dis_pred_forward(self.fake_S_logits.detach())

            loss_fake_pred_dis = 0.5 * (self.gan_loss(fake_S_seg_is_fake, False) + self.gan_loss(fake_T_seg_is_real, True))
            loss_pred_dis += loss_fake_pred_dis

            log_losses["dis/loss_fake_pred_dis"] = loss_fake_pred_dis.detach()

            if self.config['training']['lambda_adv_aux_tp'] > 0 and self.config['training']['lambda_aux_seg'] > 0:
                fake_T_aux_seg_is_real = self.dis_aux_pred_forward(self.fake_T_aux_logits.detach())
                fake_S_aux_seg_is_fake = self.dis_aux_pred_forward(self.fake_S_aux_logits.detach())

                loss_fake_aux_pred_dis = 0.5 * (self.gan_loss(fake_S_aux_seg_is_fake, False) + self.gan_loss(fake_T_aux_seg_is_real, True))
                loss_pred_dis += loss_fake_aux_pred_dis

                log_losses["dis/loss_fake_aux_pred_dis"] = loss_fake_aux_pred_dis.detach()
        
        if loss_pred_dis.item() != 0:
            loss_pred_dis.backward()
        self.visual.plot_current_errors(log_losses, self.iter)

    def step(self):
        self.set_requires_grad([self.decoder_S_DP, self.decoder_T_DP, self.D_S_DP, self.D_T_DP, self.D_P_DP, self.D_P_latent_DP], requires_grad=False)
        self.optimizer_seg.zero_grad()
        self.seg_model_backward()
        self.optimizer_seg.step()
        self.lr_scheduler_seg.step()

        if self.config['training']['lambda_s2t_seg'] > 0:
            self.set_requires_grad([self.decoder_S_DP, self.decoder_T_DP], requires_grad=True)
            self.optimizer_decoder.zero_grad()
            self.dec_backward()
            self.optimizer_decoder.step()
            self.lr_scheduler_decoder.step()
        
            self.set_requires_grad([self.D_S_DP, self.D_T_DP], requires_grad=True)
            self.optimizer_img_dis.zero_grad()
            self.img_dis_backward()
            self.optimizer_img_dis.step()
            self.lr_scheduler_img_dis.step()
        
        if self.iter > self.config['training']['start_adv_p']:
            self.set_requires_grad([self.D_P_DP, self.D_P_latent_DP], requires_grad=True)
            self.optimizer_pred_dis.zero_grad()
            self.pred_dis_backward()
            self.optimizer_pred_dis.step()
            self.lr_scheduler_img_dis.step()

    def seg_forward(self, input, labels=None, hard_anchor_sampling=False, random_sampleing=False):
        out_dict = self.seg_net_DP(input, labels, hard_anchor_sampling, random_sampleing)
        return out_dict
    
    def dec_img_toS_forward(self, multi_lv_feats):
        translated_img = self.decoder_S_DP(multi_lv_feats)
        return translated_img
    
    def dec_img_toT_forward(self, multi_lv_feats):
        translated_img = self.decoder_T_DP(multi_lv_feats)
        return translated_img
    
    def dis_img_T_forward(self, input):
        output = self.D_T_DP(input)
        return output
    
    def dis_img_S_forward(self, input):
        output = self.D_S_DP(input)
        return output
    
    def dis_pred_forward(self, prediction):
        output = self.D_P_DP(F.softmax(prediction, dim=1))
        return output
    
    def dis_aux_pred_forward(self, latent_prediction):
        output = self.D_P_latent_DP(F.softmax(latent_prediction, dim=1))
        return output

    def init_nets(self):
        # self.seg_net = DilateResUNetCLMem(n_channels=self.config["size_C"], n_classes=self.config["n_class"], 
        #     norm=self.config['seg_model']["normlization"], act=self.config['seg_model']["activation"])
        self.seg_net = DilateResUNetMultiLvSeg(self.config)
        self.multi_lv_projs = MultiProjectionHead(self.config['seg_model']['enc_multi_lv_dims'], self.config['training']['proj_dim'])
        self.decoder_T = ResUNetDecoder(self.config["size_C"], norm=self.config['decoder']["normlization"], 
            act=self.config['decoder']["activation"])
        self.decoder_S = ResUNetDecoder(self.config["size_C"], norm=self.config['decoder']["normlization"], 
            act=self.config['decoder']["activation"])
        self.D_T = Discriminator(in_ch=self.config["size_C"], norm_type=self.config['discriminator']["normlization"])
        self.D_S = Discriminator(in_ch=self.config["size_C"], norm_type=self.config['discriminator']["normlization"])
        self.D_P = DomainClassifier(in_ch=self.config["n_class"], norm_type=self.config['domain_classifier']["normlization"])
        self.D_P_latent = DomainClassifier(in_ch=self.config["n_class"], norm_type=self.config['domain_classifier']["normlization"])

        self.nets = [self.seg_net, self.decoder_T, self.decoder_S, self.D_T, self.D_S, self.D_P, self.D_P_latent]

        for net in self.nets:
            net_class_name = net.__class__.__name__
            self.logger.info("init {} with kaiming".format(net_class_name))
            if net_class_name == "DilateResUNet":
                self.init_weights(net, init_type=self.config['seg_model']['init']["init_type"], a=self.config['seg_model']['init']["init_a"])
            elif net_class_name == "DilateResUNetCLMem":
                self.init_weights(net, init_type=self.config['seg_model']['init']["init_type"], a=self.config['seg_model']['init']["init_a"])
            elif net_class_name == "DilateResUNetMultiLvSeg":
                self.init_weights(net, init_type=self.config['seg_model']['init']["init_type"], a=self.config['seg_model']['init']["init_a"])
            elif net_class_name == "MultiProjectionHead":
                self.init_weights(net, init_type=self.config['seg_model']['init']["init_type"], a=self.config['seg_model']['init']["init_a"])
            elif net_class_name == "ResUNetDecoder":
                self.init_weights(net, init_type=self.config['decoder']['init']["init_type"], a=self.config['decoder']['init']["init_a"])
            elif net_class_name == "Discriminator":
                self.init_weights(net, init_type=self.config['discriminator']['init']["init_type"], a=self.config['discriminator']['init']["init_a"])
            elif net_class_name == "DomainClassifier":
                self.init_weights(net, init_type=self.config['domain_classifier']['init']["init_type"], a=self.config['domain_classifier']['init']["init_a"])
            else:
                raise NotImplementedError("in adaptation_model init_nets function, no implemented nets")
        
        self.nets = [self.seg_net, self.multi_lv_projs, self.decoder_T, self.decoder_S, self.D_T, self.D_S, self.D_P, self.D_P_latent]

        self.seg_net_DP = self.init_device(self.seg_net, gpu_id=0, whether_DP=True)
        self.multi_lv_projs_DP = self.init_device(self.multi_lv_projs, gpu_id=0, whether_DP=True)
        self.decoder_T_DP = self.init_device(self.decoder_T, gpu_id=0, whether_DP=True)
        self.decoder_S_DP = self.init_device(self.decoder_S, gpu_id=0, whether_DP=True)
        self.D_T_DP = self.init_device(self.D_T, gpu_id=0, whether_DP=True)
        self.D_S_DP = self.init_device(self.D_S, gpu_id=0, whether_DP=True)
        self.D_P_DP = self.init_device(self.D_P, gpu_id=0, whether_DP=True)
        self.D_P_latent_DP = self.init_device(self.D_P_latent, gpu_id=0, whether_DP=True)

        self.nets_DP = [self.seg_net_DP, self.multi_lv_projs_DP, self.decoder_T_DP, self.decoder_S_DP, self.D_T_DP, self.D_S_DP, self.D_P_DP, self.D_P_latent_DP]
        
    def init_weights(self, net, init_type='normal', gain=0.02, a=0.2):
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
        
        net.apply(init_func)

        # propagate to children
        for m in net.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def init_device(self, net, gpu_id=None, whether_DP=False):
        if torch.cuda.is_available():
            net = net.cuda()
            if whether_DP:
                net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        return net
    
    def init_optimizers(self):
        optimizer_seg_params = {k:v for k, v in self.config['seg_model']['optimizer'].items() if k != 'name'}
        self.optimizer_seg = torch.optim.Adam(
            self.seg_net_DP.parameters(), **optimizer_seg_params
        )
        self.optimizer_projs = torch.optim.Adam(
            self.multi_lv_projs_DP.parameters(), **optimizer_seg_params
        )

        optimizer_dec_params = {k:v for k, v in self.config['decoder']['optimizer'].items() if k != 'name'}
        self.optimizer_decoder = torch.optim.Adam(
            itertools.chain(self.decoder_S_DP.parameters(), self.decoder_T_DP.parameters()), **optimizer_dec_params
        )

        optimizer_img_dis_params = {k:v for k, v in self.config['discriminator']['optimizer'].items() if k != 'name'}
        self.optimizer_img_dis = torch.optim.Adam(
            itertools.chain(self.D_S_DP.parameters(), self.D_T_DP.parameters()), **optimizer_img_dis_params
        )

        optimizer_pred_dis_params = {k:v for k, v in self.config['domain_classifier']['optimizer'].items() if k != 'name'}
        self.optimizer_pred_dis = torch.optim.Adam(
            itertools.chain(self.D_P_DP.parameters(), self.D_P_latent_DP.parameters()), **optimizer_pred_dis_params
        )

        self.optimizers = [self.optimizer_seg, self.optimizer_projs, self.optimizer_decoder, self.optimizer_img_dis, self.optimizer_pred_dis]
    
    def init_lr_schedulers(self, epoch_batches):
        n_epoches = self.config['training']["n_epoches"]
        lr_decay_start_step = self.config['training']['lr_schedule']["lr_decay_start_step"]
        lr_min_rate = self.config['training']['lr_schedule']["lr_min_rate"]

        self.lr_scheduler_seg = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_seg, 
            lr_lambda=LambdaLROnStep(
                n_epoches * epoch_batches, 
                0, 
                lr_decay_start_step, 
                lr_min_rate).step
        )
        self.lr_scheduler_decoder = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_decoder, 
            lr_lambda=LambdaLROnStep(
                n_epoches * epoch_batches, 
                0, 
                lr_decay_start_step, 
                lr_min_rate).step
        )
        self.lr_scheduler_img_dis = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_img_dis, 
            lr_lambda=LambdaLROnStep(
                n_epoches * epoch_batches, 
                0, 
                lr_decay_start_step, 
                lr_min_rate).step
        )
        self.lr_scheduler_pred_dis = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_pred_dis, 
            lr_lambda=LambdaLROnStep(
                n_epoches * epoch_batches, 
                0, 
                lr_decay_start_step, 
                lr_min_rate).step
        )

        self.lr_schedulers = [self.lr_scheduler_seg, self.lr_scheduler_decoder, self.lr_scheduler_img_dis, self.lr_scheduler_pred_dis]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, ema_keys, labels):
        batch_size = ema_keys.shape[0]
        feat_dim = ema_keys.shape[1]

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (ema_keys.shape[2], ema_keys.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()

        for bs in range(batch_size):
            this_feat = ema_keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != 5 and (this_label == x).nonzero().shape[0] > 9]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                # feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                # ptr = int(segment_queue_ptr[lb])
                # segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                # segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % self.memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                pixel_update_freq = self.pixel_update_freq_bg if lb == 0 else self.pixel_update_freq_fg
                K = min(num_pixel, pixel_update_freq)
                feat = this_feat[:, idxs[perm[:K]]].squeeze()
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.config['training']['pixel_memory_size']:
                    self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.config['training']['pixel_memory_size']

    @torch.no_grad()
    def _image_in_pool(self, images, labels):
        batch_size, _, H, W = images.shape
        arg_labels = torch.argmax(labels, dim=-1)

        for bs in range(batch_size):
            this_arg_labels = arg_labels[bs]
            this_arg_labels_unique = torch.unique(this_arg_labels)
            this_arg_labels_unique = [x for x in this_arg_labels_unique if x != 5 and (this_arg_labels == x).nonzero().shape[0] > 9]
            
            for lb in this_arg_labels_unique:
                ptr = self.source_images_pool_ptr[lb] % self.config['training']['image_memory_size']
                self.source_images_pool[lb, ptr] = images[bs]
                self.source_labels_pool[lb, ptr] = arg_labels[bs]
                self.source_images_pool_ptr[lb] += 1

    @torch.no_grad()
    def get_emb_queue_from_imgmem(self):
        image_samples = self.config['training']['image_samples']
        pool_images = None
        pool_labels = None

        # construct input images: randomly select n_image_samples from each category pool
        for lb in range(self.config['n_class']):
            ptr = self.source_images_pool_ptr[lb]
            real_img_mem_size = self.config['training']['image_memory_size'] if (ptr // self.config['training']['image_memory_size']) > 0 else ptr
            perm = torch.randperm(real_img_mem_size)
            if lb == 0:
                pool_images = self.source_images_pool[lb, perm[:image_samples]].cuda()
                pool_labels = self.source_labels_pool[lb, perm[:image_samples]].cuda()
            else:
                pool_images = torch.cat([pool_images, self.source_images_pool[lb, perm[:image_samples]].cuda()])
                pool_labels = torch.cat([pool_labels, self.source_labels_pool[lb, perm[:image_samples]].cuda()])
        
        out_dict = self.seg_forward(pool_images, pool_labels, hard_anchor_sampling=False, random_sampleing=True)
        # multi_lv_feats_proj = self.multi_lv_projs_DP(out_dict['multi_lv_features'])

        # extract embeddings from projection feauture map
        # emb_queue_, emb_queue_y = self.anchor_sampling(multi_lv_feats_proj, pool_labels)
        emb_queue_, emb_queue_y = out_dict['multi_lv_cat_proj_keys'], out_dict['multi_lv_cat_feats_labels_']

        return emb_queue_, emb_queue_y
    
    @torch.no_grad()
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
    
    @torch.no_grad()
    def _get_all_res_cat_anchors(self, indices, multi_lv_feat):
        # recurve to select multi lv pixels from corresponding feature maps
        cur_lv_indices = indices
        selected_pixels_cat = None
        for i, this_lv_feat in enumerate(multi_lv_feat):
            this_lv_feat_permute = this_lv_feat.reshape(this_lv_feat.shape[0], -1).transpose(0, 1)

            if type(selected_pixels_cat) == type(None):
                selected_pixels_cat = this_lv_feat_permute[cur_lv_indices].squeeze(1)
            else:
                selected_pixels_cat = torch.cat([selected_pixels_cat, this_lv_feat_permute[cur_lv_indices].squeeze(1)], dim=1)
            
            cur_lv_indices = self._get_scaled_location(cur_lv_indices, this_lv_feat.shape[1], this_lv_feat.shape[2], 2)

        return selected_pixels_cat
    
    @torch.no_grad()
    def anchor_sampling(self, multi_lv_feats, labels):
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

    @torch.no_grad()
    def orig_anchor_sampling(self, multi_lv_feats_proj, labels):
        pass
        # image_samples = self.config['training']['image_samples']

        # batch_size = labels.shape[0]

        # labels = labels.contiguous().view(batch_size, -1)

        # feat_dim = feats.shape[-1]

        # X_ = None
        # y_ = None

        # # for cls_i, cls_lb in enumerate(range(self.config['n_class'])):
        # #     this_y_hat = labels[cls_i * image_samples: (cls_i + 1) * image_samples]
        # #     lb_indices = (this_y_hat == cls_lb).nonzero()

        # #     max_views = 10 if cls_id == 0 else self.config['training']['max_views']
        # #     n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

        # #     perm = torch.randperm(n_indices)
        # #     selected_embedding_indices = indices[perm[:n_sel_pixels_clas_id]]

        # #     if cls_i == 0:
        # #         X_ = feats[cls_i * image_samples: (cls_i + 1) * image_samples, selected_embedding_indices, :].squeeze()
        # #         y_ = torch.zeros(n_sel_pixels_clas_id, dtype=torch.long, device="cuda:0").fill_(cls_id)
        # #     pass

        # for ii in range(batch_size):
        #     this_y_hat = labels[ii]
        #     this_classes = torch.unique(this_y_hat)
        #     this_classes = [x for x in this_classes if x != 5 and (this_y_hat == x).nonzero().shape[0] > 2]

        #     for cls_i, cls_id in enumerate(this_classes):
        #         # if cls_id == 0 or cls_id == 2:
        #         #     continue
        #         indices = (this_y_hat == cls_id).nonzero()

        #         n_indices = indices.shape[0]

        #         max_views = 30 if cls_id == 0 else self.config['training']['max_views']
        #         n_sel_pixels_clas_id = min(n_indices, max_views)  # cls_id pixels selected in this image

        #         perm = torch.randperm(n_indices)
        #         selected_embedding_indices = indices[perm[:n_sel_pixels_clas_id]]

        #         # print()
        #         # print("anchor_sampling function feats[ii, selected_embedding_indices, :].squeeze() shape:", feats[ii, selected_embedding_indices, :].squeeze().shape)
        #         # print("-------------------------")

        #         if type(X_) == type(None) and type(y_) == type(None):
        #             X_ = torch.cat([feats[ii, selected_embedding_indices, :].squeeze(), feats[ii, indices, :].squeeze().mean(dim=0, keepdim=True)])
        #             y_ = torch.zeros(n_sel_pixels_clas_id + 1, dtype=torch.long, device="cuda:0").fill_(cls_id)
        #         else:
        #             X_ = torch.cat([X_, feats[ii, selected_embedding_indices, :].squeeze(), feats[ii, indices, :].squeeze().mean(dim=0, keepdim=True)], dim=0)
        #             y_ = torch.cat([y_, torch.zeros(n_sel_pixels_clas_id + 1, dtype=torch.long, device="cuda:0").fill_(cls_id)])

        # return X_, y_

    def get_pseodu_labels(self, logits):
        with torch.no_grad():
            logits_detach = logits.detach()
            logits_detach_softmax = logits_detach.softmax(dim=1)
            tmp_logits_clone = logits_detach_softmax.clone()
            logits_detach_softmax[tmp_logits_clone >= 0.5] = 10.0
            _tmp = np.full([self.config['data']['target']["batch_size"], 1, self.config["size_W"], self.config["size_H"]], 1.0,)
            _tmp = torch.Tensor(_tmp).cuda()
            _tmp = torch.cat((logits_detach_softmax, _tmp), 1)
            pseudo_label = _tmp.argmax(1)
            pseudo_label_one_hot = one_hot(pseudo_label, self.config['n_class']+1).permute(0,2,3,1).detach()
        return pseudo_label_one_hot

    def fake_images_pool(self, num_fakes, fake, fake_pool):
        if num_fakes < self.config['training']["pool_size"]:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self.config['training']["pool_size"] - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake
    
    def detach(self, input):
        if self.config['training']['lambda_detach_cyc']:
            if isinstance(input, list):
                detach_output = []
                for e in input:
                    detach_output.append(e.detach())
                return detach_output
            else:
                return input.detach()
        else:
            return input

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def train(self):
        for net in self.nets:
            net.train()
        for net in self.nets_DP:
            net.train()
    
    def eval(self):
        for net in self.nets:
            net.eval()
        for net in self.nets_DP:
            net.eval()
    
    @torch.no_grad()
    def visualization(self):
        visual_s_list = []
        visual_t_list = []

        # visual_s_list += [('real_S', self.source_images.cpu()), 
        #                   ('label_seg_S', self.source_labels.argmax(-1).data.cpu().numpy()),
        #                   ('pred_seg_S', self.source_out_dict['seg'].softmax(1).argmax(1).data.cpu().numpy())
        #                 ]
        # visual_t_list += [('real_T', self.target_images.cpu()), 
        #                   ('label_seg_T', self.target_labels.argmax(-1).data.cpu().numpy()),
        #                   ('pred_seg_T', self.target_out_dict['seg'].softmax(1).argmax(1).data.cpu().numpy())
        #                 ]

        if self.config['training']['tsne']:
            # self.tsne_vis(self.latent_pixel_queue)
            # self.tsne_vis(self.out_pixel_queue)
            self.tsne_vis(self.pixel_queue)

        if self.config['training']['lambda_s2t_seg'] > 0:
            if self.config['training']["lambda_consis_pred"] and self.iter > self.config['training']["consis_pred_start_step"]:
                visual_t_list.append(('target_pseudo_seg', self.target_pseudo_label.argmax(-1).data.cpu().numpy()))
            
            visual_s_list += [('fake_T', self.fake_T.cpu()), 
                              ('pred_seg_fake_T', self.fake_T_logits.softmax(1).argmax(1).data.cpu().numpy())
                            ]
            visual_t_list += [('fake_S', self.fake_S.cpu()), 
                              ('pred_seg_fake_S', self.fake_S_logits.softmax(1).argmax(1).data.cpu().numpy())
                            ]
            
            if self.config['training']["lambda_cycle_t"] > 0 and self.config['training']["lambda_cycle_s"] > 0:
                visual_s_list.append(('cyc_S', self.cyc_S.cpu()))
                visual_t_list.append(('cyc_T', self.cyc_T.cpu()))
            
            if self.config['training']["lambda_id_t"] > 0 and self.config['training']["lambda_id_s"] > 0:
                visual_s_list.append(('id_S', self.id_S.cpu()))
                visual_t_list.append(('id_T', self.id_T.cpu()))
        
        # self.visual.display_current_results(OrderedDict(visual_s_list), "s", self.iter)
        # self.visual.display_current_results(OrderedDict(visual_t_list), "t", self.iter)
    
    @torch.no_grad()
    def tsne_vis(self, queue):
        n_class, memory_size, dims = queue.shape
        perm = torch.randperm(memory_size)
        samples = self.config['training']['tsne_samples']
        feats = np.zeros((self.config['n_class'] * samples, dims))

        for lb in range(self.config['n_class']):
            feats[lb * samples : (lb + 1) * samples, :] = queue[lb, perm[:samples], :].cpu().numpy()
        
        X_embedded = tsnecuda.TSNE().fit_transform(feats)
        em_x = X_embedded[:, 0]
        em_y = X_embedded[:, 1]

        scale_01_em_x = self.scale_to_01_range(em_x)
        scale_01_em_y = self.scale_to_01_range(em_y)

        # initialize a matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        contour_map = {
            0: "bg",
            1: "la_myo",
            2: "la_blood",
            3: "lv_blood",
            4: "aa",
        }

        colors_label = np.array([
            [0,     0,   0],
            [254, 232,  81], #LV-myo
            [145, 193,  62], #LA-blood
            [ 29, 162, 220], #LV-blood
            [238,  37,  36]]) #AA

        # for every class, we'll add a scatter plot separately
        for label in range(self.config['n_class']):
            # convert the class color to matplotlib format
            color = np.array(colors_label[label], dtype=np.float).reshape((1, -1)) / 255

            # add a scatter plot with the corresponding color and label
            ax.scatter(scale_01_em_x[label*samples:(label+1)*samples], scale_01_em_y[label*samples:(label+1)*samples], 
                        s=7, c=color, label=contour_map[label])

        # build a legend using the labels we set previously
        ax.legend(loc='best')

        # finally, show the plot
        os.makedirs(os.path.join(self.logdir, "tsne", str(dims)), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, "tsne", str(dims), "{}.png".format(str(self.iter))))
        
    def scale_to_01_range(self, x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    @torch.no_grad()
    def val_step(self, val_dataloader):
        self.eval()
        torch.cuda.empty_cache()

        val_mem_size = 15000
        val_embed_queue = torch.zeros((self.config['n_class'], val_mem_size, self.config['training']['proj_dim']), dtype=torch.float32, device="cuda:0", requires_grad=False)
        val_embed_queue_ptr = torch.zeros(self.config['n_class'], dtype=torch.long, requires_grad=False)
        
        for val_batch in val_dataloader:
            val_images, val_labels, _ = val_batch
            val_images = val_images.cuda()
            val_labels = val_labels.cuda().argmax(-1)

            # forward
            val_out_dict = self.seg_forward(val_images)
            # print()
            keys = val_out_dict['keys']
            batch_size = keys.shape[0]
            feat_dim = keys.shape[1]

            ds_labels = val_labels[:, ::self.network_stride, ::self.network_stride]  # 网络下采样了8倍，所以label也下采样，只是这里用间隔采样代替了

            for bs in range(batch_size):
                this_feat = keys[bs].contiguous().view(feat_dim, -1)
                this_label = ds_labels[bs].contiguous().view(-1)
                this_label_ids = torch.unique(this_label)
                this_label_ids = [x for x in this_label_ids if x != 5]

                for lb in this_label_ids:
                    # print("enque and deque, lb is", lb)
                    idxs = (this_label == lb).nonzero()

                    # pixel enqueue and dequeue
                    num_pixel = idxs.shape[0]
                    perm = torch.randperm(num_pixel)
                    K = min(num_pixel, self.pixel_update_freq)
                    feat = this_feat[:, perm[:K]]
                    feat = torch.transpose(feat, 0, 1)

                    ptr = int(val_embed_queue_ptr[lb])

                    if ptr + K >= val_mem_size:
                        val_embed_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                        val_embed_queue_ptr[lb] = 0
                    else:
                        # print()
                        # print("_dequeue_and_enqueue function K value is", K, "feat shape is ", feat.shape)
                        # print("-----------------------------")
                        val_embed_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                        # fix the bug that the pixel queue ptr update only 1 step instead of K steps
                        val_embed_queue_ptr[lb] = (val_embed_queue_ptr[lb] + K) % val_mem_size
        
        self.tsne_vis_val(val_embed_queue, val_embed_queue_ptr)

    @torch.no_grad()
    def val_step(self, val_dataloader):
        self.eval()
        torch.cuda.empty_cache()

        val_mem_size = 15000
        val_embed_queue = torch.zeros((self.config['n_class'], val_mem_size, self.config['training']['proj_dim']), dtype=torch.float32, device="cuda:0", requires_grad=False)
        val_embed_queue_ptr = torch.zeros(self.config['n_class'], dtype=torch.long, requires_grad=False)
        
        for val_batch in val_dataloader:
            val_images, val_labels, _ = val_batch
            val_images = val_images.cuda()
            val_labels = val_labels.cuda().argmax(-1)

            # forward
            val_out_dict = self.seg_forward(val_images)
            # print()
            keys = val_out_dict['keys']
            batch_size = keys.shape[0]
            feat_dim = keys.shape[1]

            # ds_labels = val_labels[:, ::self.network_stride, ::self.network_stride]  # 网络下采样了8倍，所以label也下采样，只是这里用间隔采样代替了
            val_labels = val_labels.unsqueeze(1).float().clone()
            val_labels = torch.nn.functional.interpolate(val_labels,
                                                    (keys.shape[2], keys.shape[3]), mode='nearest')
            val_labels = val_labels.squeeze(1).long()

            for bs in range(batch_size):
                this_feat = keys[bs].contiguous().view(feat_dim, -1)
                this_label = val_labels[bs].contiguous().view(-1)
                this_label_ids = torch.unique(this_label)
                this_label_ids = [x for x in this_label_ids if x != 5]

                for lb in this_label_ids:
                    # print("enque and deque, lb is", lb)
                    idxs = (this_label == lb).nonzero()

                    pixel_update_freq = self.pixel_update_freq_bg if lb == 0 else self.pixel_update_freq_fg

                    # pixel enqueue and dequeue
                    num_pixel = idxs.shape[0]
                    perm = torch.randperm(num_pixel)
                    K = min(num_pixel, pixel_update_freq)
                    feat = this_feat[:, perm[:K]]
                    feat = torch.transpose(feat, 0, 1)

                    ptr = int(val_embed_queue_ptr[lb])

                    if ptr + K >= val_mem_size:
                        val_embed_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                        val_embed_queue_ptr[lb] = 0
                    else:
                        # print()
                        # print("_dequeue_and_enqueue function K value is", K, "feat shape is ", feat.shape)
                        # print("-----------------------------")
                        val_embed_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                        # fix the bug that the pixel queue ptr update only 1 step instead of K steps
                        val_embed_queue_ptr[lb] = (val_embed_queue_ptr[lb] + K) % val_mem_size
        
        self.tsne_vis_val(val_embed_queue, val_embed_queue_ptr)

    @torch.no_grad()
    def tsne_vis_val(self, queue, queue_ptr):
        n_class, _, dims = queue.shape
        samples = self.config['training']['tsne_samples']
        feats = []

        selected_samples = dict()
        for lb in range(self.config['n_class']):
            # if lb == 0 or lb == 2:
            #     continue

            lb_embed_count = queue_ptr[lb]

            if lb_embed_count >= samples: # saved embeddings is enough, then randomly selects n_samples
                perm = torch.randperm(lb_embed_count)
                feats.append(queue[lb, perm[:samples], :].cpu().numpy())
                selected_samples[lb] = samples
            else:  # saved embeddings less than required samples, then selects all saved embeddings
                feats.append(queue[lb, :lb_embed_count, :].cpu().numpy())
                selected_samples[lb] = lb_embed_count
        feats = np.concatenate(feats)

        X_embedded = tsnecuda.TSNE().fit_transform(feats)
        em_x = X_embedded[:, 0]
        em_y = X_embedded[:, 1]

        scale_01_em_x = self.scale_to_01_range(em_x)
        scale_01_em_y = self.scale_to_01_range(em_y)

        # initialize a matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        contour_map = {
            0: "bg",
            1: "la_myo",
            2: "la_blood",
            3: "lv_blood",
            4: "aa",
        }

        colors_label = np.array([
            [0,     0,   0],
            [254, 232,  81], #LV-myo
            [145, 193,  62], #LA-blood
            [ 29, 162, 220], #LV-blood
            [238,  37,  36]]) #AA

        # for every class, we'll add a scatter plot separately
        feat_ptr = 0
        for label in range(self.config['n_class']):
            # if label == 0 or label == 2:
            #     continue
            # convert the class color to matplotlib format
            color = np.array(colors_label[label], dtype=np.float).reshape((1, -1)) / 255

            # add a scatter plot with the corresponding color and label
            ax.scatter(scale_01_em_x[feat_ptr:feat_ptr + selected_samples[label]], 
                        scale_01_em_y[feat_ptr:feat_ptr + selected_samples[label]], 
                        s=7, c=color, label=contour_map[label])
            feat_ptr += selected_samples[label]

        # build a legend using the labels we set previously
        ax.legend(loc='best')

        # finally, show the plot
        os.makedirs(os.path.join(self.logdir, "tsne_val", str(dims)), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, "tsne_val", str(dims), "{}.png".format(str(self.iter))))

    def save(self):
        pass


