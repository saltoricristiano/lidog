import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from utils.models.minkunet import MinkUNet34
from utils.models.minkunet_ibn import MinkUNet34IBN
from utils.models.minkunet_robustnet import MinkUNet34Robust
from utils.models.minkunet_bev import MinkUNet34BEV
from utils.datasets.initialization import get_dataset
from utils.datasets.dataset import MultiSourceDataset
from configs import get_config
from utils.collation import CollateFN
from utils.pipelines import PLTTrainer
from utils.pipelines import PLTTrainerBEV


parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/semantickitti.yaml",
                    type=str,
                    help="Path to config file")
parser.add_argument("--resume_checkpoint",
                    default=None,
                    help="If not provided in configs, ckpt to be evaluated")
parser.add_argument("--save_predictions",
                    action='store_true',
                    default=False,
                    help="Save or not predictions")


def evaluate(config):

    def get_dataloader(dataset, batch_size, collate_fn, shuffle=False, pin_memory=True):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)

    def get_model(config):
        if config.model.name == 'MinkUNet34':
            m = MinkUNet34(in_channels=config.model.in_channels,
                           out_channels=config.model.out_channels,
                           D=config.model.D,
                           initial_kernel_size=config.model.conv1_kernel_size,
                           )
        elif config.model.name == 'MinkUNet34IBN':
            m = MinkUNet34IBN(in_channels=config.model.in_channels,
                              out_channels=config.model.out_channels,
                              D=config.model.D,
                              initial_kernel_size=config.model.conv1_kernel_size)
        elif config.model.name == 'MinkUNet34Robust':
            m = MinkUNet34Robust(in_channels=config.model.in_channels,
                                 out_channels=config.model.out_channels,
                                 D=config.model.D,
                                 initial_kernel_size=config.model.conv1_kernel_size)
        elif config.model.name == 'MinkUNet34BEV':
            bottle_img_dim = dict(zip(config.model.decoder_2d_levels, config.model.bev_feats_sizes))
            bottle_out_img_dim = dict(zip(config.model.decoder_2d_levels, config.model.bev_img_sizes))

            try:
                scaling_factors = dict(zip(config.model.decoder_2d_levels, config.model.scaling_factors))
            except AttributeError:
                scaling_factors = {'block8': 1.0, 'block7': 1.0, 'block6': 1.0, 'bottle': 1.0}

            try:
                binary_segmentation_layer = config.model.binary_segmentation_layer
            except:
                binary_segmentation_layer = False

            m = MinkUNet34BEV(in_channels=config.model.in_channels,
                              out_channels=config.model.out_channels,
                              D=config.model.D,
                              initial_kernel_size=config.model.conv1_kernel_size,
                              decoder_2d_level=config.model.decoder_2d_levels,
                              bottle_img_dim=bottle_img_dim,
                              bottle_out_img_dim=bottle_out_img_dim,
                              scaling_factors=scaling_factors,
                              binary_seg_layer=binary_segmentation_layer)
        else:
            raise NotImplementedError
        print(f'--> Using {config.model.name}!')
        return m

    def get_source_domains():
        training_dataset = []
        validation_dataset = []

        num_source_domains = len(config.source_dataset.name)

        for sd in range(len(config.source_dataset.name)):
            dataset_name = config.source_dataset.name[sd]
            training_dataset_tmp, validation_dataset_tmp = get_dataset(dataset_name=dataset_name,
                                                                       voxel_size=config.source_dataset.voxel_size,
                                                                       sub_p=config.source_dataset.sub_p,
                                                                       num_classes=config.model.out_channels,
                                                                       ignore_label=config.source_dataset.ignore_label,
                                                                       use_cache=config.source_dataset.use_cache,
                                                                       augmentation_list=config.source_dataset.augmentation_list)

            training_dataset.append(training_dataset_tmp)
            validation_dataset.append(validation_dataset_tmp)

        if num_source_domains == 1:
            training_dataset = training_dataset[0]
            validation_dataset = validation_dataset[0]

        else:
            training_dataset = MultiSourceDataset(training_dataset)

        return training_dataset, validation_dataset

    def get_target_domains():

        num_target_domains = len(config.target_dataset.name)

        if num_target_domains == 2:

            target_dataset = []
            for td in range(len(config.target_dataset.name)):
                dataset_name = config.target_dataset.name[td]
                _, target_dataset_tmp = get_dataset(dataset_name=dataset_name,
                                                    voxel_size=config.target_dataset.voxel_size,
                                                    sub_p=config.target_dataset.sub_p,
                                                    num_classes=config.model.out_channels,
                                                    ignore_label=config.target_dataset.ignore_label,
                                                    use_cache=config.target_dataset.use_cache,
                                                    augmentation_list=config.target_dataset.augmentation_list)

                target_dataset.append(target_dataset_tmp)

        elif num_target_domains == 1:
            dataset_name = config.target_dataset.name[0]
            _, target_dataset = get_dataset(dataset_name=dataset_name,
                                            voxel_size=config.target_dataset.voxel_size,
                                            sub_p=config.target_dataset.sub_p,
                                            num_classes=config.model.out_channels,
                                            ignore_label=config.target_dataset.ignore_label,
                                            use_cache=config.target_dataset.use_cache,
                                            augmentation_list=config.target_dataset.augmentation_list)

        else:
            raise NotImplementedError

        return target_dataset

    model = get_model(config)

    training_dataset, validation_dataset = get_source_domains()

    collation_single = CollateFN()

    target_dataset = get_target_domains()

    if len(config.target_dataset.name) > 1:
        target_dataloader = [get_dataloader(t_dataset, collate_fn=collation_single, batch_size=config.pipeline.dataloader.batch_size*2, shuffle=False) for t_dataset in target_dataset]
    else:
        target_dataloader = get_dataloader(target_dataset,
                                           collate_fn=collation_single,
                                           batch_size=config.pipeline.dataloader.batch_size*2,
                                           shuffle=False)

    if config.pipeline.lightning.resume_checkpoint is not None:
        resume_from_checkpoint = config.pipeline.lightning.resume_checkpoint
    elif args.resume_checkpoint is not None:
        resume_from_checkpoint = args.resume_checkpoint
    else:
        raise AttributeError('You must provide a checkpoint for evaluation!')

    ckpt_dir, _ = os.path.split(resume_from_checkpoint)
    save_dir, _ = os.path.split(ckpt_dir)
    _, run_name = os.path.split(save_dir)
    run_name = run_name + '_EVALUATION'

    save_preds_dir = os.path.join(save_dir, 'predictions')

    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=run_name,
                               offline=True)

    loggers = [wandb_logger]

    strategy = None
    if config.model.name in ['MinkUNet34BEV']:
        pl_module = PLTTrainerBEV(training_dataset=training_dataset,
                                  validation_dataset=validation_dataset,
                                  model=model,
                                  sem_criterion=config.pipeline.losses.sem_criterion,
                                  optimizer_name=config.pipeline.optimizer.name,
                                  batch_size=config.pipeline.dataloader.batch_size,
                                  val_batch_size=config.pipeline.dataloader.batch_size,
                                  lr=config.pipeline.optimizer.lr,
                                  num_classes=config.model.out_channels,
                                  train_num_workers=config.pipeline.dataloader.num_workers,
                                  val_num_workers=config.pipeline.dataloader.num_workers,
                                  clear_cache_int=config.pipeline.lightning.clear_cache_int,
                                  scheduler_name=config.pipeline.scheduler.name,
                                  source_domains_name=config.source_dataset.name,
                                  target_domains_name=config.target_dataset.name,
                                  save_dir=save_dir,
                                  save_predictions=args.save_predictions,
                                  save_folder=save_preds_dir)
    else:
        pl_module = PLTTrainer(training_dataset=training_dataset,
                               validation_dataset=validation_dataset,
                               model=model,
                               sem_criterion=config.pipeline.losses.sem_criterion,
                               optimizer_name=config.pipeline.optimizer.name,
                               batch_size=config.pipeline.dataloader.batch_size,
                               val_batch_size=config.pipeline.dataloader.batch_size,
                               lr=config.pipeline.optimizer.lr,
                               num_classes=config.model.out_channels,
                               train_num_workers=config.pipeline.dataloader.num_workers,
                               val_num_workers=config.pipeline.dataloader.num_workers,
                               clear_cache_int=config.pipeline.lightning.clear_cache_int,
                               scheduler_name=config.pipeline.scheduler.name,
                               source_domains_name=config.source_dataset.name,
                               target_domains_name=config.target_dataset.name,
                               save_dir=save_dir,
                               save_predictions=args.save_predictions,
                               save_folder=save_preds_dir)

    trainer = Trainer(max_epochs=config.pipeline.epochs,
                      gpus=config.pipeline.gpus,
                      strategy=strategy,
                      default_root_dir=config.pipeline.save_dir,
                      precision=config.pipeline.precision,
                      logger=loggers,
                      check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                      val_check_interval=config.pipeline.lightning.val_check_interval,
                      num_sanity_val_steps=config.pipeline.lightning.num_sanity_val_steps,
                      log_every_n_steps=50)

    trainer.test(pl_module,
                 ckpt_path=resume_from_checkpoint,
                 dataloaders=target_dataloader)


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True

    evaluate(config)
