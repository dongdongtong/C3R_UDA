import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import one_hot


class _SoftWeightedSoftmaxLoss(nn.Module):
    """
        Calculate soft teacher weighted cross-entropy loss.
    """
    def __init__(self, num_classes=5, reduction="mean"):
        super(_SoftWeightedSoftmaxLoss, self).__init__()
        self.num_classes = num_classes if num_classes > 1 else num_classes + 1

    def forward(self, logits, gt, ema_logits):
        softmaxpred = torch.softmax(logits, dim=1)
        ema_softmaxpred = torch.softmax(ema_logits, dim=1)

        raw_loss = 0
        for i in range(self.num_classes):
            gti = gt[..., i]
            predi = softmaxpred[:, i, :, :]
            
            weighted = 1 - (torch.sum(gti) / torch.sum(gt[:, :, :, :-1])).item()
            soft_teacher_weight = ema_softmaxpred[:, i, :, :]

            if i == 0:
                raw_loss = -1.0 * weighted * soft_teacher_weight * gti * torch.log(torch.clamp(predi, 0.005, 1))
            else:
                raw_loss += -1.0 * weighted * soft_teacher_weight * gti * torch.log(torch.clamp(predi, 0.005, 1))

        loss = torch.mean(raw_loss)

        return loss


class _SoftDiceLoss(nn.Module):
    """
        Calculate dice loss.
    """
    def __init__(self, num_classes=5, eps=1e-7):
        super(_SoftDiceLoss, self).__init__()
        self.num_classes = num_classes if num_classes > 1 else num_classes + 1
        self.eps = eps

    def forward(self, logits, gt):
        dice = 0
        softmaxpred = torch.softmax(logits, dim=1)

        for i in range(self.num_classes):
            inse = torch.sum(softmaxpred[:, i, :, :] * gt[:, i, :, :])
            l = torch.sum(softmaxpred[:, i, :, :] * softmaxpred[:, i, :, :])
            r = torch.sum(gt[:, i, :, :] * gt[:, i, :, :])
            dice += 2.0 * inse / (l + r + self.eps)

        # print("DiceLoss shape:", dice)

        return 1 - 1.0 * dice / self.num_classes


class SoftTaskLoss(nn.Module):
    """
        Calculate task loss, which consists of the weighted cross entropy loss and dice loss
    """
    def __init__(self, num_classes=5):
        super(SoftTaskLoss, self).__init__()
        self.ce_loss = _SoftWeightedSoftmaxLoss(num_classes=num_classes)
        self.dice_loss = _SoftDiceLoss(num_classes=num_classes)
        self.num_classes = num_classes if num_classes > 1 else num_classes + 1

    def forward(self, logits, gt, ema_soft_mask, ema_logits):
        # print("TaskLoss", logits.shape, logits.device, gt.shape, gt.device)
        return self.ce_loss(logits, gt, ema_soft_mask, ema_logits), self.dice_loss(logits, gt, ema_soft_mask, ema_logits)