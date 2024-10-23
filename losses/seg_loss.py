import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import one_hot


class _WeightedSoftmaxLoss(nn.Module):
    """
        Calculate weighted cross-entropy loss.
    """
    def __init__(self, num_classes=5, reduction="mean"):
        super(_WeightedSoftmaxLoss, self).__init__()
        self.num_classes = num_classes if num_classes > 1 else num_classes + 1

    def forward(self, logits, gt):
        # softmaxpred = torch.clamp(torch.softmax(logits, dim=1), 0.005, 1)
        # return F.nll_loss(torch.log(softmaxpred), gt, None, None, ignore_index=5, reduce=None, reduction="mean")

        # softmaxpred = torch.softmax(logits, dim=1)
        # if (gt.unique() == 5).sum() > 0:
        #     gt_one_hot = one_hot(gt, self.num_classes+1).detach()
        #     gt_sum = gt_one_hot[:, :-1, :, :].sum()
        # else:
        #     gt_one_hot = one_hot(gt, self.num_classes).detach()
        #     gt_sum = gt_one_hot.sum()
        
        # raw_loss = 0
        # for i in range(self.num_classes):
        #     gti = gt_one_hot[:, i, :, :]
        #     predi = softmaxpred[:, i, :, :]
        #     weighted = 1 - (torch.sum(gti) / gt_sum).item()

        #     if i == 0:
        #         raw_loss = -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))
        #     else:
        #         raw_loss += -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))

        # loss = torch.mean(raw_loss)

        # return loss

        softmaxpred = torch.softmax(logits, dim=1)
        # print("_SoftmaxWeightedLoss softmaxpred shape:", softmaxpred.shape, softmaxpred.device)

        raw_loss = 0
        for i in range(self.num_classes):
            gti = gt[..., i]
            predi = softmaxpred[:, i, :, :]
            weighted = 1 - (torch.sum(gti) / torch.sum(gt)).item()

            if i == 0:
                raw_loss = -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))
            else:
                raw_loss += -1.0 * weighted * gti * torch.log(torch.clamp(predi, 0.005, 1))

        loss = torch.mean(raw_loss)

        return loss


class _DiceLoss(nn.Module):
    """
        Calculate dice loss.
    """
    def __init__(self, num_classes=5, eps=1e-7):
        super(_DiceLoss, self).__init__()
        self.num_classes = num_classes if num_classes > 1 else num_classes + 1
        self.eps = eps

    def forward(self, logits, gt):
        # logits = logits.permute((0, 2, 3, 1))
        # dice = 0
        # softmaxpred = torch.softmax(logits, dim=1)

        # # gt_one_hot = one_hot(gt, self.num_classes)
        # if (gt.unique() == 5).sum() > 0:
        #     gt_one_hot = one_hot(gt, self.num_classes+1).detach()
        # else:
        #     gt_one_hot = one_hot(gt, self.num_classes).detach()

        # for i in range(self.num_classes):
        #     inse = torch.sum(softmaxpred[:, i, :, :] * gt_one_hot[:, i, :, :])
        #     l = torch.sum(softmaxpred[:, i, :, :] * softmaxpred[:, i, :, :])
        #     r = torch.sum( gt_one_hot[:, i, :, :])
        #     dice += 2.0 * inse / (l + r + self.eps)

        # # print("DiceLoss shape:", dice)

        # return 1 - 1.0 * dice / self.num_classes

        dice = 0
        softmaxpred = torch.softmax(logits, dim=1)

        for i in range(self.num_classes):
            inse = torch.sum(softmaxpred[:, i, :, :] * gt[:, :, :, i])
            l = torch.sum(softmaxpred[:, i, :, :] * softmaxpred[:, i, :, :])
            r = torch.sum(gt[:, :, :, i])
            dice += 2.0 * inse / (l + r + self.eps)

        # print("DiceLoss shape:", dice)

        return 1 - 1.0 * dice / self.num_classes


class TaskLoss(nn.Module):
    """
        Calculate task loss, which consists of the weighted cross entropy loss and dice loss
    """
    def __init__(self, num_classes=5):
        super(TaskLoss, self).__init__()
        self.ce_loss = _WeightedSoftmaxLoss(num_classes=num_classes)
        self.dice_loss = _DiceLoss(num_classes=num_classes)
        self.num_classes = num_classes if num_classes > 1 else num_classes + 1

    def forward(self, logits, gt):
        # print("TaskLoss", logits.shape, logits.device, gt.shape, gt.device)
        return self.ce_loss(logits, gt), self.dice_loss(logits, gt)