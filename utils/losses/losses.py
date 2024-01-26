import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class CELoss(nn.Module):
    def __init__(self, ignore_label: int = None, weight: np.ndarray = None):
        '''
        :param ignore_label: label to ignore
        :param weight: possible weights for weighted CE Loss
        '''
        super().__init__()
        if weight is not None:
            weight = torch.from_numpy(weight).float()
            print(f'----->Using weighted CE Loss weights: {weight}')

        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_label, weight=weight)
        self.ignored_label = ignore_label

    def forward(self, preds: torch.Tensor, gt: torch.Tensor):

        loss = self.loss(preds, gt)
        return loss


class SoftCELoss(nn.Module):
    def __init__(self, dim=-1, ignore_index=None):
        super(SoftCELoss, self).__init__()
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        b, h, w, c = target.shape

        pred = pred.view(b*h*w, c).cpu()
        target = target.view(b*h*w, c).cpu()

        if self.ignore_index:
            valid_idx = torch.logical_not(target[:, 0] == -1)
            pred = pred[valid_idx]
            target = target[valid_idx]

        nan_idx = torch.logical_not(torch.isnan(target.sum(dim=-1)))
        pred = pred[nan_idx]
        target = target[nan_idx]

        pred = pred.log_softmax(dim=self.dim)
        loss = torch.mean(torch.sum(-target * pred, dim=self.dim))
        if loss > 100:
            print(loss)
        return loss


class DICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=False, use_tmask=False):
        super(DICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label

        self.powerize = powerize
        self.use_tmask = use_tmask

    def forward(self, output, target):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target = F.one_hot(target, num_classes=output.shape[1])
        output = F.softmax(output, dim=-1)

        intersection = (output * target).sum(dim=0)
        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)

        dice_loss = 1 - iou.mean()

        return dice_loss.to(input_device)


def get_soft(t_vector, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    return t_soft


def get_kitti_soft(t_vector, labels, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    searched_idx = torch.logical_or(labels == 6, labels == 1)
    if searched_idx.sum() > 0:
        t_soft[searched_idx, 1] = max_val/2
        t_soft[searched_idx, 6] = max_val/2

    return t_soft


class SoftDICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=True,
                 neg_range=False, eps=0.05, is_kitti=False):
        super(SoftDICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.neg_range = neg_range
        self.eps = eps
        self.is_kitti = is_kitti

    def forward(self, output, target, return_class=False, is_kitti=False):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target_onehot = F.one_hot(target, num_classes=output.shape[1])
        if not self.is_kitti and not is_kitti:
            target_soft = get_soft(target_onehot, eps=self.eps)
        else:
            target_soft = get_kitti_soft(target_onehot, target, eps=self.eps)

        output = F.softmax(output, dim=-1)

        intersection = (output * target_soft).sum(dim=0)

        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target_onehot.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target_onehot.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)
        iou_class = tmask * 2 * intersection / union

        if self.neg_range:
            dice_loss = -iou.mean()
            dice_class = -iou_class
        else:
            dice_loss = 1 - iou.mean()
            dice_class = 1 - iou_class
        if return_class:
            return dice_loss.to(input_device), dice_class
        else:
            return dice_loss.to(input_device)


class SoftLabelDICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=False,
                 neg_range=False):
        super(SoftLabelDICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.neg_range = neg_range

    def forward(self, output, target, return_class=False):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        b, h, w, c = target.shape

        target = target.view(b*h*w, c)
        output = output.view(b*h*w, c)

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target[:, 0] == self.ignore_label)
            target = target[valid_idx, :]
            output = output[valid_idx, :]

        output = F.softmax(output, dim=-1)

        intersection = (output * target).sum(dim=0)

        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)
        iou_class = tmask * 2 * intersection / union

        if self.neg_range:
            dice_loss = -iou.mean()
            dice_class = -iou_class
        else:
            dice_loss = 1 - iou.mean()
            dice_class = 1 - iou_class
        if return_class:
            return dice_loss.to(input_device), dice_class
        else:
            return dice_loss.to(input_device)


class WMSELoss(nn.Module):
    def __init__(self):
        super(WMSELoss, self).__init__()

    def __call__(self, output, target, weights=None):

        out = (output - target) ** 2
        # if out.ndim > 1:
        #     out = out.sum(dim=1)
        if weights is not None:
            out = out * weights.expand_as(out)
        loss = out.mean(0)
        # loss = F.mse_loss(output, target.view(-1), reduction='mean')
        return loss


class PDFNormal(nn.Module):

    def __init__(self):
        super(PDFNormal, self).__init__()

    def __call__(self, x, mean, var):
        """
        Computes instance belonging probability values
        :param x: embeddings values of all points NxD
        :param mean: instance embedding 1XD
        :param var: instance variance value 1XD
        :return: probability scores for all points Nx1
        """
        eps = torch.ones_like(var, requires_grad=True, device=x.device) * 1e-8
        var_eps = var + eps
        var_seq = var_eps.squeeze()
        inv_var = torch.diag(1 / var_seq)
        mean_rep = mean.repeat(x.shape[0], 1)
        dif = x - mean_rep
        d = torch.pow(dif, 2)
        e = torch.matmul(d, inv_var)
        probs = torch.exp(e * -0.5)
        probs = torch.sum(probs, 1) / torch.sum(var_eps)

        return probs


class IoUInstanceLoss(nn.Module):
    def __init__(self, range_th: float = 0.9):
        super(IoUInstanceLoss, self).__init__()
        self.new_pdf_normal = PDFNormal()
        self.weighted_mse_loss = WMSELoss()
        self.range_th = range_th

    def __call__(self, centers_p, embeddings, variances, ins_labels, points=None, times=None, th=None):
        """
        Computes l2 loss between gt-prob values and predicted prob values for instances
        :param centers_p: objectness scores Nx1
        :param embeddings: embeddings  NxD
        :param variances: variances NxD
        :param ins_labels: instance ids Nx1
        :param points: xyz values Nx3
        :param times: time value normalized between 0-1 Nx1
        :return: instance loss
        """
        if th is None:
            th = self.range_th
        instances = torch.unique(ins_labels)
        loss = torch.tensor(0.0).to(embeddings.device)
        loss.requires_grad = True

        # if variances.shape[1] - embeddings.shape[1] > 4:
        #     global_emb, _ = torch.max(embeddings, 0, keepdim=True)
        #     embeddings = torch.cat((embeddings, global_emb.repeat(embeddings.shape[0],1)),1)

        if variances.shape[1] - embeddings.shape[1] == 3:
            embeddings = torch.cat((embeddings, points), 1)
        if variances.shape[1] - embeddings.shape[1] == 4:
            embeddings = torch.cat((embeddings, points, times), 1)

        for instance in instances:
            if instance != 0:
                ins_idxs = torch.where(ins_labels == instance)[0]
                ins_centers = centers_p[ins_idxs]
                sorted, indices = torch.sort(ins_centers, 0, descending=True)
                range = torch.sum(sorted > th)
                if range == 0:
                    random_center = 0
                else:
                    random_center = torch.randint(0, range, (1,)).item()

                idx = ins_idxs[indices[random_center]]
                mean = embeddings[idx]  # 1xD

                var = variances[idx]

                labels = (ins_labels == instance) * 1.0

                probs = self.new_pdf_normal(embeddings, mean, var)

                ratio = torch.sum(ins_labels == 0)/(torch.sum(ins_labels == instance)*1.0+ torch.sum(probs > 0.5))
                weights = ((ins_labels.view(-1) == instance) | (probs >0.5)) * ratio + (ins_labels.view(-1) >= 0) * 1 #new loss
                loss = loss + self.weighted_mse_loss(probs, labels.view(-1), weights)
                # loss = loss + self.weighted_mse_loss(probs, labels.view(-1))

        return loss


class VarianceSmoothLoss(nn.Module):
    def __init__(self):
        super(VarianceSmoothLoss, self).__init__()

    def __call__(self, variances, ins_labels):
        """
        Computes smoothness loss between variance predictions
        :param variances: variances NxD
        :param ins_labels: instance ids Nx1
        :return: variance loss
        """
        instances = torch.unique(ins_labels)
        loss = torch.tensor(0.0).to(variances.device)
        loss.requires_grad = True
        if instances.size()[0] == 1:
            return torch.tensor(0)
        for instance in instances:
            if instance == 0:
                continue
            ins_idxs = torch.where(ins_labels == instance)
            ins_variances = variances[ins_idxs]
            var = torch.mean(ins_variances, dim=0)
            var_gt = var.repeat(ins_variances.shape[0], 1)
            ins_loss = torch.nn.MSELoss()(ins_variances, var_gt)

            loss = loss + ins_loss
            if torch.isnan(ins_loss):
                print("nan")
        return loss


class KLVAELoss(nn.Module):
    def __init__(self):
        super(KLVAELoss, self).__init__()

    def __call__(self, means, log_vars):
        return -0.5 * torch.mean(torch.mean(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1))


# class FocalLoss(nn.Module):
#
#     def __init__(self, gamma=2, alpha=0.25, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, output, target):
#         target = target.view(-1, 1)
#         target = target.long()
#         logpt = F.logsigmoid(output)
#         logpt = logpt.gather(0, target)
#         logpt = logpt.view(-1)
#         pt = logpt.exp()
#
#         if self.alpha is not None:
#             if self.alpha.type() != output.data.type():
#                 self.alpha = self.alpha.type_as(output.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * at
#
#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class IRWLoss(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, f_map, eye, mask_matrix, margin, num_remove_cov):
        f_cor, BN = self.get_covariance_matrix(f_map, eye=eye)
        f_cor_masked = f_cor * mask_matrix

        off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1, 2), keepdim=True) - margin # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
        loss = torch.sum(loss)/BN

        return loss

    def get_covariance_matrix(self, f_map, eye=None):
        eps = 1e-5
        BN, C = f_map.shape  # i-th feature size (B X C X H X W)
        if eye is None:
            eye = torch.eye(C).cuda()
        f_map_resh = f_map.view(BN, C, -1)  # B X C X H X W > B X C X (H X W)
        f_cor = torch.bmm(f_map_resh, f_map_resh.transpose(1, 2)).div(BN-1) + (eps * eye)  # C X C / HW

        return f_cor, BN


class IWLoss(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, f_map, eye, mask_matrix, margin, num_remove_cov):
        f_cor, BN = self.get_covariance_matrix(f_map, eye=eye)
        f_cor_masked = f_cor * mask_matrix

        off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1, 2), keepdim=True)
        loss = torch.sum(off_diag_sum)/BN

        return loss

    def get_covariance_matrix(self, f_map, eye=None):
        eps = 1e-5
        BN, C = f_map.shape  # i-th feature size (B X C X H X W)
        if eye is None:
            eye = torch.eye(C).cuda()
        f_map_resh = f_map.view(BN, C, -1)  # B X C X H X W > B X C X (H X W)
        f_cor = torch.bmm(f_map_resh, f_map_resh.transpose(1, 2)).div(BN-1) + (eps * eye)  # C X C / HW

        return f_cor, BN
