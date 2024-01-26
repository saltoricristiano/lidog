import os
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from utils.losses import *
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score
import open3d as o3d


class PLTTrainerBEV(pl.core.LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """

    def __init__(self,
                 model,
                 training_dataset,
                 validation_dataset,
                 optimizer_name='SGD',
                 sem_criterion='WCELoss',
                 aux_criterion='SoftDICELoss',
                 warmup_epochs=10,
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
                 save_predictions=False,
                 save_folder=None):

        super().__init__()
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.optimizer_name = optimizer_name
        self.sem_criterion = sem_criterion
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
        self.save_predictions = save_predictions
        self.save_folder = save_folder

        self.source_domains = source_domains_name
        self.target_domains = target_domains_name

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

        if self.aux_criterion == 'SoftLabelDICELoss':
            self.sem_bev_criterion = SoftLabelDICELoss(ignore_label=self.training_dataset.ignore_label)
        elif self.aux_criterion == 'SoftCELoss':
            self.sem_bev_criterion = SoftCELoss(ignore_index=self.training_dataset.ignore_label)
        else:
            self.sem_bev_criterion = CELoss(ignore_label=self.training_dataset.ignore_label)

    def training_step(self, batch, batch_idx):
        source0 = self.source_domains[0]

        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        stensor = ME.SparseTensor(coordinates=batch["source_coordinates0"].int(), features=batch["source_features0"])

        out0, bev_preds0 = self.model(stensor)

        sem_preds0 = out0.F
        sem_labels0 = batch['source_sem_labels0']
        bev_sem_labels0 = batch['source_bev_map0']

        b, h, w, c = bev_sem_labels0.shape

        if self.aux_criterion == 'CELoss':
            invalid_bev_labels0 = bev_sem_labels0[:, :, :, 0] == -1
            bev_sem_labels0 = bev_sem_labels0.argmax(dim=-1)
            bev_sem_labels0[invalid_bev_labels0] = -1
            sem_loss_bev0 = self.sem_bev_criterion(bev_preds0.view(b, c, h * w), bev_sem_labels0.view(b, h*w))
        else:
            sem_loss_bev0 = self.sem_bev_criterion(bev_preds0, bev_sem_labels0)

        # semantic segmentation
        sem_loss0 = self.sem_criterion(sem_preds0, sem_labels0).cpu()

        total_loss = self.source_weights[0] * sem_loss0 + self.aux_weights[0] * sem_loss_bev0

        # sem_loss1 = self.sem_criterion(sem_preds1, sem_labels1).cpu()
        # total_loss = self.source_weights[0] * sem_loss0 + self.source_weights[1] * sem_loss1
        # total_loss += self.aux_weights[0] * aux_loss0 + self.aux_weights[1] * aux_loss1
        #
        _, preds0 = sem_preds0.max(1)
        iou_tmp0 = jaccard_score(preds0.detach().cpu().numpy(), sem_labels0.cpu().numpy(), average=None,
                                 labels=np.arange(0, self.num_classes),
                                 zero_division=0.)

        # domain 0
        present_labels, class_occurs = np.unique(sem_labels0.cpu().numpy(), return_counts=True)
        class_occurs = class_occurs[present_labels != self.ignore_label]
        present_labels = present_labels[present_labels != self.ignore_label]
        names = self.training_dataset.class2names[present_labels+1].tolist()
        present_names = [os.path.join('training', source0, p + '_iou') for p in names]
        occurrences = [os.path.join('training', source0, p + '_count') for p in names]
        results_dict = dict(zip(present_names, iou_tmp0.tolist()))
        occurs_dict = dict(zip(occurrences, class_occurs.tolist()))
        results_dict.update(occurs_dict)

        results_dict[f'training/{source0}/sem_loss0'] = sem_loss0.item()
        results_dict[f'training/{source0}/bev_loss0'] = sem_loss_bev0.item()
        results_dict[f'training/{source0}/source_iou0'] = np.mean(iou_tmp0[present_labels])
        results_dict['training/lr'] = self.trainer.optimizers[0].param_groups[0]["lr"]
        results_dict['training/epoch'] = self.current_epoch
        results_dict[f'training/{source0}/total_loss'] = total_loss.item()

        bev_argmax_preds0 = bev_preds0.detach().cpu().argmax(dim=1)
        if self.aux_criterion in ['SoftCELoss']:
            invalid_bev_labels0 = bev_sem_labels0[:, :, :, 0] == -1
            bev_argmax_labels0 = bev_sem_labels0.argmax(dim=-1)
            bev_argmax_labels0[invalid_bev_labels0] = -1
            bev_argmax_labels0 = bev_argmax_labels0.cpu()
        else:
            bev_argmax_labels0 = bev_sem_labels0.cpu()

        iou_bev_tmp0 = jaccard_score(bev_argmax_preds0.view(-1).numpy(), bev_argmax_labels0.view(-1).numpy(), average=None,
                                 labels=np.arange(0, self.num_classes),
                                 zero_division=0.)
        present_labels_bev = torch.unique(bev_argmax_labels0)
        present_labels_bev = present_labels_bev[present_labels_bev != self.ignore_label]

        results_dict[f'training/{source0}/bev_source_iou0'] = np.mean(iou_bev_tmp0[present_labels_bev])

        if self.global_step % 1000 == 0 and self.current_epoch > 5:

            os.makedirs(os.path.join(self.save_dir, 'bev_images', 'gt'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'bev_images', 'preds'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'bev_images', 'dense'), exist_ok=True)

            bev_sem_img0 = bev_argmax_labels0[0]
            invalid_pixels = bev_sem_img0 == -1
            bev_image_gt0 = self.training_dataset.color_map[bev_sem_img0+1]

            bev_pred_img0 = bev_argmax_preds0[0]
            bev_pred_dense_img0 = bev_pred_img0.clone()
            bev_pred_img0[invalid_pixels] = -1
            bev_image_pred0 = self.training_dataset.color_map[bev_pred_img0+1]

            # we find the square around scan
            valid_pixels = np.logical_not(invalid_pixels)
            x_vals, y_vals = np.where(valid_pixels)
            x_min = x_vals.min()
            x_max = x_vals.max()
            y_min = y_vals.min()
            y_max = y_vals.max()
            bev_pred_dense_img0[:x_min, :] = -1
            bev_pred_dense_img0[x_max:, :] = -1
            bev_pred_dense_img0[:, :y_min] = -1
            bev_pred_dense_img0[:, y_max:] = -1

            bev_image_dense0 = self.training_dataset.color_map[bev_pred_dense_img0+1]

            plt.imsave(os.path.join(self.save_dir, 'bev_images', 'gt',
                                    str(self.current_epoch)+str(self.global_step)+'_0.jpg'),
                       bev_image_gt0)
            plt.imsave(os.path.join(self.save_dir, 'bev_images', 'preds',
                                    str(self.current_epoch)+str(self.global_step)+'_0.jpg'),
                       bev_image_pred0)

            plt.imsave(os.path.join(self.save_dir, 'bev_images', 'dense',
                                    str(self.current_epoch)+str(self.global_step)+'_0.jpg'),
                       bev_image_dense0)

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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        phase = self.target_domains[dataloader_idx]

        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])

        out, _ = self.model(stensor)
        out = out.F

        sem_preds = out
        sem_labels = batch["sem_labels"].long()

        _, preds = sem_preds.max(1)
        iou_tmp = jaccard_score(preds.detach().cpu().numpy(), sem_labels.cpu().numpy(), average=None,
                                 labels=np.arange(0, self.num_classes),
                                 zero_division=0.)

        present_labels, class_occurs = np.unique(sem_labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]

        iou_tmp = torch.from_numpy(iou_tmp)

        iou = -torch.ones_like(iou_tmp)
        iou[present_labels] = iou_tmp[present_labels]

        if self.save_predictions:
            coords = batch["coordinates"].cpu()
            labels = batch["sem_labels"].long().cpu()
            preds = preds.cpu()

            batch_size = torch.unique(coords[:, 0]).max() + 1
            sample_idx = batch["idx"]
            for b in range(batch_size.int()):
                s_idx = int(sample_idx[b].item())
                b_idx = coords[:, 0] == b
                points = coords[b_idx, 1:]
                p = preds[b_idx]
                l = labels[b_idx]

                valid_idx = l != self.ignore_label
                points = points[valid_idx]
                p = p[valid_idx]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(self.training_dataset.color_map[p+1])

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(self.training_dataset.color_map[p+1])

                os.makedirs(os.path.join(self.save_folder, phase, 'preds'), exist_ok=True)
                o3d.io.write_point_cloud(os.path.join(self.save_folder, phase, 'preds', f'{s_idx}.ply'), pcd)

        return {'iou': iou,
                'domain': phase}

    def test_epoch_end(self, outputs):

        if self.num_source_domains > 1:
            source_names = ''
            for s in range(len(self.source_domains)):
                source_names += self.source_domains[s]
        else:
            source_names = self.source_domains[0]

        if self.num_target_domains > 1:
            target_names = ''
            for t in range(len(self.target_domains)):
                target_names += self.target_domains[t]
        else:
            target_names = self.target_domains[0]

        csv_file = os.path.join(self.save_dir, 'results', source_names + '-TO-' + target_names + '.csv')
        try:
            os.makedirs(os.path.join(self.save_dir, 'results'))
        except OSError:
            print('--> Overwriting results !!')
            os.makedirs(os.path.join(self.save_dir, 'results'), exist_ok=True)

        csv_columns = ['source', 'target']

        for c in self.training_dataset.class2names[1:]:
            csv_columns.append(c)

        csv_columns.append('mean')
        if len(self.target_domains) == 1:
            outputs = [outputs]

        for o in range(len(outputs)):
            mean_iou = []
            for return_dict in outputs[o]:
                # get predictions
                iou_tmp = return_dict["iou"]
                target_name = return_dict["domain"]

                nan_idx = iou_tmp == -1
                iou_tmp[nan_idx] = float('nan')
                mean_iou.append(iou_tmp.unsqueeze(0))

            mean_iou = torch.cat(mean_iou, dim=0).numpy()
            per_class_iou = np.nanmean(mean_iou, axis=0) * 100
            average_iou = np.nanmean(per_class_iou, axis=0)

            with open(csv_file, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if o == 0:
                    writer.writerow(csv_columns)

                results_row = [source_names, target_name]

                for p in per_class_iou:
                    results_row.append(str(round(p, 2)).replace('.', ','))

                results_row.append(str(round(float(average_iou), 2)).replace('.', ','))
                writer.writerow(results_row)

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
