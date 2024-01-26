import os
import MinkowskiEngine as ME
from utils.losses import *
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score
from torchmetrics import JaccardIndex


class PLTTrainer2DMulti(pl.core.LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """

    def __init__(self,
                 model,
                 training_dataset,
                 validation_dataset,
                 optimizer_name='SGD',
                 sem_criterion='WCELoss',
                 sem_bev_criterion='CELoss',
                 aux_criterion='KLDivLoss',
                 warmup_epochs=1,
                 lr=1e-3,
                 batch_size=12,
                 weight_decay=1e-4,
                 momentum=0.98,
                 val_batch_size=6,
                 train_num_workers=10,
                 val_num_workers=10,
                 num_classes=7,
                 clear_cache_int=2,
                 scheduler_name=None,
                 source_weights=[1., 1.],
                 aux_weights=[0.5],
                 source_domains_name=None,
                 target_domains_name=None,
                 save_dir=None,
                 individual_pred=False,
                 log_bev_3d_iou=False):

        super().__init__()
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.optimizer_name = optimizer_name
        self.sem_criterion = sem_criterion
        self.sem_bev_criterion = sem_bev_criterion
        self.aux_criterion = aux_criterion
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.val_batch_size = val_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.num_classes = num_classes
        self.clear_cache_int = clear_cache_int
        self.scheduler_name = scheduler_name
        self.source_weights = source_weights
        self.aux_weights = aux_weights
        self.ignore_label = self.training_dataset.ignore_label
        self.save_dir = save_dir
        self.individual_pred = individual_pred
        self.log_bev_3d_iou = log_bev_3d_iou

        self.source_domains = source_domains_name
        self.target_domains = target_domains_name

        self.iou_metric = JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_label, average='none')

        self.num_source_domains = len(self.source_domains)

        if self.target_domains is not None:
            self.num_target_domains = len(self.target_domains)
        else:
            self.num_target_domains = 0

        self.init_losses()

        self.save_hyperparameters(ignore='model')

    def init_losses(self):
        if self.sem_criterion == 'CELoss':
            self.sem_criterion = CELoss(ignore_label=self.training_dataset.ignore_label,
                                    weight=None)

        elif self.sem_criterion == 'DICELoss':
            self.sem_criterion = DICELoss(ignore_label=self.training_dataset.ignore_label)

        elif self.sem_criterion == 'SoftDICELoss':
            if self.num_classes == 19:
                self.sem_criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label, is_kitti=True)
            else:
                self.sem_criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        elif self.sem_criterion == 'FocalLoss':
            self.sem_criterion = FocalLoss(alpha=0.25,
                                           gamma=2)
        else:
            raise NotImplementedError

        if self.sem_bev_criterion == 'SoftLabelDICELoss':
            self.sem_bev_criterion = SoftLabelDICELoss(ignore_label=self.training_dataset.ignore_label)
        elif self.sem_bev_criterion == 'SoftCELoss':
            self.sem_bev_criterion = SoftCELoss(ignore_index=self.training_dataset.ignore_label)
        elif self.sem_bev_criterion == 'SoftDICELoss':
            self.sem_bev_criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        elif self.sem_bev_criterion == 'DICELoss':
            self.sem_bev_criterion = DICELoss(ignore_label=self.training_dataset.ignore_label)
        else:
            self.sem_bev_criterion = CELoss(ignore_label=self.training_dataset.ignore_label)

        if self.aux_criterion == 'KLDivLoss':
            self.aux_criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
        elif self.aux_criterion == 'DICELoss':
            self.aux_criterion = DICELoss()
        else:
            raise NotImplementedError

    @staticmethod
    def select_3d(preds, labels, indices, batch_idx, batch_size=4):
        concat_preds = []
        concat_labels = []
        for b in range(batch_size+1):
            batch_pts_idx = batch_idx == b
            batch_bev_idx = torch.logical_not(indices[b] == -1)
            batch_bev_idx = indices[b, batch_bev_idx]
            batch_preds = preds[batch_pts_idx]
            batch_labels = labels[batch_pts_idx]
            concat_preds.append(batch_preds[batch_bev_idx])
            concat_labels.append(batch_labels[batch_bev_idx])

        return torch.cat(concat_preds, dim=0), torch.cat(concat_labels, dim=0)

    def training_step(self, batch, batch_idx):

        # name of the domain
        source0 = self.source_domains[0]
        source1 = self.source_domains[1]

        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        # get sparse tensor
        stensor0 = ME.SparseTensor(coordinates=batch["source_coordinates0"].int(), features=batch["source_features0"])
        stensor1 = ME.SparseTensor(coordinates=batch["source_coordinates1"].int(), features=batch["source_features1"])

        # get batch idx
        batch_idx_3d0 = batch["source_coordinates0"][:, 0]
        batch_idx_3d1 = batch["source_coordinates1"][:, 0]

        # get bev labels
        bev_sem_labels0 = batch['source_bev_labels0']
        bev_sem_labels1 = batch['source_bev_labels1']

        # get bev labels
        bev_sampled_idx0 = batch['source_bev_sampled_idx0']
        bev_sampled_idx1 = batch['source_bev_sampled_idx1']

        # get semantic labels
        sem_labels0 = batch['source_sem_labels0']
        sem_labels1 = batch['source_sem_labels1']

        # predict
        out0, bev_preds0 = self.model(stensor0, is_train=True)
        out1, bev_preds1 = self.model(stensor1, is_train=True)

        # get prediction features
        sem_preds0 = out0.F
        sem_preds1 = out1.F

        # init bev loss
        sem_loss_bev0 = torch.tensor(0.)
        sem_loss_bev1 = torch.tensor(0.)

        # for logging also the dict
        loss_dict_bev = {}

        # iterate over decoders
        for key in bev_sem_labels0.keys():
            # get decoder key loss
            sem_loss_bev0_tmp = self.sem_bev_criterion(bev_preds0[key].view(-1, self.num_classes).cpu(),
                                                       bev_sem_labels0[key].view(-1).cpu())

            sem_loss_bev1_tmp = self.sem_bev_criterion(bev_preds1[key].view(-1, self.num_classes).cpu(),
                                                       bev_sem_labels1[key].view(-1).cpu())

            # sum divided by tot number of decoders
            sem_loss_bev0 += sem_loss_bev0_tmp / len(bev_preds0.keys())
            sem_loss_bev1 += sem_loss_bev1_tmp / len(bev_preds1.keys())

            # for logging
            loss_dict_bev[f'training/{source0}/source_bev0_loss_{key}'] = sem_loss_bev0_tmp.detach().item()
            loss_dict_bev[f'training/{source1}/source_bev1_loss_{key}'] = sem_loss_bev1_tmp.detach().item()

        if self.current_epoch >= self.warmup_epochs:
            sem_loss0 = self.sem_criterion(sem_preds0, sem_labels0).cpu()
            sem_loss1 = self.sem_criterion(sem_preds1, sem_labels1).cpu()

            total_loss0 = sem_loss0 + sem_loss_bev0
            total_loss1 = sem_loss1 + sem_loss_bev1
            total_loss = self.source_weights[0] * total_loss0 + self.source_weights[1] * total_loss1
        else:
            sem_loss0 = torch.tensor(0.)
            sem_loss1 = torch.tensor(0.)
            # aux_loss0 = torch.tensor(0.)
            # aux_loss1 = torch.tensor(0.)

            total_loss = self.source_weights[0] * sem_loss_bev0 + self.source_weights[1] * sem_loss_bev1

        # logging of pts
        # get predictions
        pts_argmax_preds0 = F.softmax(sem_preds0, dim=-1).detach().argmax(dim=-1)
        pts_argmax_preds1 = F.softmax(sem_preds1, dim=-1).detach().argmax(dim=-1)

        # filter valid labels (we have -1 ignored)
        valid_idx_pts0 = torch.logical_not(sem_labels0 == -1)
        valid_idx_pts1 = torch.logical_not(sem_labels1 == -1)

        # get IoU per class
        iou_pts_tmp0 = self.iou_metric(pts_argmax_preds0.view(-1)[valid_idx_pts0], sem_labels0.view(-1)[valid_idx_pts0]).cpu()
        iou_pts_tmp1 = self.iou_metric(pts_argmax_preds1.view(-1)[valid_idx_pts1], sem_labels1.view(-1)[valid_idx_pts1]).cpu()

        # get present labels and filter -1
        present_labels_pts0, class_occurs_pts0 = torch.unique(sem_labels0.cpu(), return_counts=True)
        present_labels_pts0 = present_labels_pts0[present_labels_pts0 != self.ignore_label]

        present_labels_pts1, class_occurs_pts1 = torch.unique(sem_labels1.cpu(), return_counts=True)
        present_labels_pts1 = present_labels_pts1[present_labels_pts1 != self.ignore_label]

        # get str names
        names_pts0 = self.training_dataset.class2names[present_labels_pts0+1].tolist()
        present_names_pts0 = [os.path.join('training', source0, p + '_iou') for p in names_pts0]
        results_dict0 = dict(zip(present_names_pts0, iou_pts_tmp0.tolist()))

        names_pts1 = self.training_dataset.class2names[present_labels_pts1+1].tolist()
        present_names_pts1 = [os.path.join('training', source1, p + '_iou') for p in names_pts1]
        results_dict1 = dict(zip(present_names_pts1, iou_pts_tmp1.tolist()))

        # bev preds can be multiple level -> IoU for each level -> log each level

        for lvl in bev_sem_labels0.keys():
            # get level labels
            bev_argmax_labels0 = bev_sem_labels0[lvl]
            # get level preds
            b, h, w = bev_argmax_labels0.shape

            bev_argmax_preds0 = bev_preds0[lvl].detach().view(b, h, w, -1).argmax(dim=-1)

            # filter invalid labels
            valid_idx_bev0 = torch.logical_not(bev_argmax_labels0.view(-1) == -1)

            # get IoU
            iou_bev_tmp0 = self.iou_metric(bev_argmax_preds0.view(-1)[valid_idx_bev0], bev_argmax_labels0.view(-1)[valid_idx_bev0]).cpu()

            # get present classes
            present_labels_bev0 = torch.unique(bev_argmax_labels0.cpu())
            present_labels_bev0 = present_labels_bev0[present_labels_bev0 != self.ignore_label]

            results_dict_bev_lvl0 = dict()
            results_dict_bev_lvl0[f'training/{source0}/source_iou_bev0_{lvl}'] = torch.mean(iou_bev_tmp0[present_labels_bev0])

            results_dict0.update(results_dict_bev_lvl0)

        for lvl in bev_sem_labels1.keys():
            # get level labels
            bev_argmax_labels1 = bev_sem_labels1[lvl]
            # get level preds
            b, h, w = bev_argmax_labels1.shape

            bev_argmax_preds1 = bev_preds1[lvl].detach().view(b, h, w, -1).argmax(dim=-1)

            # filter invalid labels
            valid_idx_bev1 = torch.logical_not(bev_argmax_labels1.view(-1) == -1)

            # get IoU
            iou_bev_tmp1 = self.iou_metric(bev_argmax_preds1.view(-1)[valid_idx_bev1], bev_argmax_labels1.view(-1)[valid_idx_bev1]).cpu()

            # get present classes
            present_labels_bev1 = torch.unique(bev_argmax_labels1.cpu())
            present_labels_bev1 = present_labels_bev1[present_labels_bev1 != self.ignore_label]

            results_dict_bev_lvl1 = dict()
            results_dict_bev_lvl1[f'training/{source1}/source_iou_bev1_{lvl}'] = torch.mean(iou_bev_tmp1[present_labels_bev1])

            results_dict1.update(results_dict_bev_lvl1)

        results_dict = dict()
        results_dict.update(results_dict0)
        results_dict.update(results_dict1)

        results_dict[f'training/{source0}/sem_loss0'] = sem_loss0.item()
        results_dict[f'training/{source0}/bev_loss0'] = sem_loss_bev0.item()
        results_dict[f'training/{source0}/source_iou0'] = torch.mean(iou_pts_tmp0[present_labels_pts0])

        results_dict[f'training/{source1}/sem_loss1'] = sem_loss1.item()
        results_dict[f'training/{source1}/bev_loss1'] = sem_loss_bev1.item()
        results_dict[f'training/{source1}/source_iou1'] = torch.mean(iou_pts_tmp1[present_labels_pts1])

        results_dict['training/lr'] = self.trainer.optimizers[0].param_groups[0]["lr"]
        results_dict['training/epoch'] = self.current_epoch
        results_dict[f'training/total_loss'] = total_loss.item()

        self.log_losses(results_dict)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        phase = self.source_domains[dataloader_idx]

        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        out, _ = self.model(stensor)

        sem_preds = out.F
        sem_labels = batch['sem_labels'].long()
        # semantic segmentation
        sem_loss = self.sem_criterion(sem_preds, sem_labels).cpu()

        _, preds = sem_preds.max(1)
        iou_tmp = jaccard_score(preds.detach().cpu().numpy(), sem_labels.cpu().numpy(), average=None,
                                 labels=np.arange(0, self.num_classes),
                                 zero_division=0.)

        present_labels, class_occurs = np.unique(sem_labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels+1].tolist()
        present_names = [os.path.join('validation', phase, p + '_source_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict[f'validation/{phase}/sem_loss'] = sem_loss.item()
        results_dict[f'validation/{phase}/source_iou'] = np.mean(iou_tmp[present_labels])
        results_dict['validation/epoch'] = self.current_epoch

        self.log_losses(results_dict)

        return results_dict

    def log_losses(self, results_dict, on_epoch=False):

        on_step = not on_epoch

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.batch_size)

    def configure_optimizers(self):
        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            if self.scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            elif self.scheduler_name == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            elif self.scheduler_name == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                              base_lr=self.lr/10000,
                                                              max_lr=self.lr,
                                                              step_size_up=5,
                                                              mode="triangular2",
                                                              cycle_momentum=False)

            else:
                raise NotImplementedError

            return [optimizer], [scheduler]

