import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import MinkowskiEngine as ME
from sklearn.cluster import DBSCAN

from utils.models.minkunet import MinkUNet34
from utils.models.minkunet_ibn import MinkUNet34IBN
from utils.datasets.initialization import get_dataset
from utils.datasets.sn_scaling import SingleSNSourceDataset, MultiSNSourceDataset
from configs import get_config
from utils.collation import CollateFN, CollateFNMultiSource, CollateFNSingleSource
from utils.pipelines import PLTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/semantickitti.yaml",
                    type=str,
                    help="Path to config file")
parser.add_argument("--auto_resume",
                    "-auto",
                    action='store_true',
                    default=False,
                    help="Automatically resume training from last checkpoint")


def get_average_dims(dataset, min_pts=5000, min_cluster_pts=50, min_car_pts=1000):
    avg_shape = []
    selected_idx = np.arange(len(dataset))
    selected_idx = np.random.choice(selected_idx, int(0.2 * selected_idx.shape[0]))

    if dataset.name == 'NuScenesDataset':
        min_pts = 2000
        min_car_pts = 300

    for s in selected_idx:
        data = dataset.__getitem__(s)
        pcd_tmp = data['coordinates'] * dataset.voxel_size
        lbl_tmp = data['sem_labels']

        # select only car points
        car_idx = lbl_tmp == 0
        if torch.sum(car_idx) > min_pts:
            car_pts = pcd_tmp[car_idx]

            # perform a rough clustering on dense points
            cluster_idx = DBSCAN(eps=0.5, min_samples=10).fit_predict(car_pts)

            # get count
            clusters, counts = np.unique(cluster_idx, return_counts=True)
            clusters = clusters[clusters != -1]

            for c_idx in clusters:
                cluster_pts_idx = cluster_idx == c_idx

                if np.sum(cluster_pts_idx) > min_car_pts:

                    clusted_pts = car_pts[cluster_pts_idx].numpy()

                    dim_0_min = np.min(clusted_pts[:, 0])
                    dim_0_max = np.max(clusted_pts[:, 0])

                    dim_1_min = np.min(clusted_pts[:, 1])
                    dim_1_max = np.max(clusted_pts[:, 1])

                    dim_2_min = np.min(clusted_pts[:, 2])
                    dim_2_max = np.max(clusted_pts[:, 2])

                    w = dim_0_max - dim_0_min
                    height = dim_1_max - dim_1_min
                    l = dim_2_max - dim_2_min

                    length = np.max([w, l])
                    width = np.min([w, l])

                    if 1 < width < 4 and 1 < height < 4 and 3 < length < 7:
                        avg_shape.append(np.array([width, height, length])[np.newaxis, ...])

    return np.mean(np.concatenate(avg_shape, axis=0), axis=0)


def get_scaling_params(source_datasets, target_datasets):
    os.makedirs('utils/datasets/_avg_sizes', exist_ok=True)

    # get average size of car instances for each domain()
    source_avg_shape = []
    for s_dataset in source_datasets:
        s_file_name = os.path.join('utils/datasets/_avg_sizes', s_dataset.name.lower()+'.npy')
        if not os.path.exists(s_file_name):
            s_size_tmp = get_average_dims(s_dataset)
            np.save(s_file_name, s_size_tmp)
        else:
            s_size_tmp = np.load(s_file_name)
        source_avg_shape.append(s_size_tmp)

    target_avg_shape = []
    for t_dataset in target_datasets:
        t_file_name = os.path.join('utils/datasets/_avg_sizes', t_dataset.name.lower()+'.npy')

        if not os.path.exists(t_file_name):
            t_size_tmp = get_average_dims(t_dataset)
            np.save(t_file_name, t_size_tmp)
        else:
            t_size_tmp = np.load(t_file_name)
        target_avg_shape.append(t_size_tmp)

    scaling_set = []

    for s_avg_tmp in source_avg_shape:
        scaling_tmp = []
        for t_avg_tmp in target_avg_shape:
            scale_tmp_0 = t_avg_tmp[0] / s_avg_tmp[0]
            scale_tmp_1 = t_avg_tmp[1] / s_avg_tmp[1]
            scale_tmp_2 = t_avg_tmp[2] / s_avg_tmp[2]

            scaling_tmp.append(np.array([scale_tmp_0, scale_tmp_1, scale_tmp_2])[np.newaxis, ...])

        scaling_tmp = np.concatenate(scaling_tmp, axis=0)
        scaling_set.append(scaling_tmp)

    return scaling_set


def train(config):

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
                           initial_kernel_size=config.model.conv1_kernel_size)
        elif config.model.name == 'MinkUNet34IBN':
            m = MinkUNet34IBN(in_channels=config.model.in_channels,
                              out_channels=config.model.out_channels,
                              D=config.model.D,
                              initial_kernel_size=config.model.conv1_kernel_size)

        else:
            raise NotImplementedError
        print(f'--> Using {config.model.name}!')
        return m

    def get_run_name(config):
        run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())

        run_time += config.model.name
        source_name = ''
        for s in range(len(config.source_dataset.name)):
            source_name += config.source_dataset.name[s]

        target_name = ''
        for s in range(len(config.target_dataset.name)):
            target_name += config.target_dataset.name[s]

        if config.pipeline.wandb.run_name is not None:
            run_name = run_time + source_name + '-TO-' + target_name + '_' + config.pipeline.wandb.run_name + '_'
        else:
            run_name = run_time + '_'
        run_name += 'BS' + str(config.pipeline.dataloader.batch_size) + '_'
        run_name += str(config.pipeline.optimizer.name) + '_'
        run_name += str(config.pipeline.optimizer.lr) + '_'
        run_name += str(config.pipeline.scheduler.name) + '_'
        run_name += str(config.pipeline.losses.sem_criterion) + '_'
        run_name += 'AUG' if config.source_dataset.augmentation_list is not None else 'NO_AUG'
        return run_name

    def get_source_domains():
        training_dataset = []
        validation_dataset = []

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

        return training_dataset, validation_dataset

    def get_target_domains():

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

        return target_dataset

    def get_last_checkpoint(save_path):
        # list all paths and get the last one
        if not os.path.exists(save_path):
            return None, None

        all_names = os.listdir(os.path.join(save_path))

        if len(all_names) == 0:
            return None, None
        else:
            all_dates = [n[:16] for n in all_names]
            years = [int(n[:4]) for n in all_dates]
            months = [int(n[5:7]) for n in all_dates]
            days = [int(n[8:10]) for n in all_dates]
            h = [int(n[11:13]) for n in all_dates]
            m = [int(n[14:16]) for n in all_dates]
            last_idx = np.argmax(np.array(years) * 365 * 24 * 60 + np.array(months) * 30 * 24 * 60 + np.array(days) * 24 * 60 + np.array(h) * 60 + np.array(m))
            last_path = all_names[last_idx]

            # among all checkpoints we need to find the last
            all_ckpt = os.listdir(os.path.join(save_path, last_path, "checkpoints"))
            ep = [e[6:8] for e in all_ckpt]
            ckpts = []
            for e in ep:
                if not e.endswith("-"):
                    ckpts.append(int(e))
                else:
                    ckpts.append(int(e[0]))
            last_idx = np.argmax(np.array(ckpts))

            return os.path.join(save_path, last_path, "checkpoints", all_ckpt[last_idx]), last_path

    model = get_model(config)

    training_dataset, validation_dataset = get_source_domains()
    target_dataset = get_target_domains()

    scaling_paramenters = get_scaling_params(training_dataset, target_dataset)

    if len(training_dataset) == 1:
        training_dataset = training_dataset[0]
        validation_dataset = validation_dataset[0]

        training_dataset = SingleSNSourceDataset(training_dataset, scaling_paramenters)
    else:
        training_dataset = MultiSNSourceDataset(training_dataset, scaling_paramenters)

    collation_single = CollateFN()
    collation_source = CollateFNMultiSource() if isinstance(training_dataset, MultiSNSourceDataset) else CollateFNSingleSource()

    training_dataloader = get_dataloader(training_dataset,
                                         collate_fn=collation_source,
                                         batch_size=config.pipeline.dataloader.batch_size,
                                         shuffle=True)

    if len(config.source_dataset.name) > 1:
        validation_dataloader = [get_dataloader(v_dataset, collate_fn=collation_single, batch_size=config.pipeline.dataloader.batch_size, shuffle=False) for v_dataset in validation_dataset]
    else:
        validation_dataloader = get_dataloader(validation_dataset,
                                               collate_fn=collation_single,
                                               batch_size=config.pipeline.dataloader.batch_size,
                                               shuffle=False)

    if len(config.target_dataset.name) > 1:
        target_dataloader = [get_dataloader(t_dataset, collate_fn=collation_single, batch_size=config.pipeline.dataloader.batch_size*2, shuffle=False) for t_dataset in target_dataset]
    else:
        target_dataloader = get_dataloader(target_dataset,
                                           collate_fn=collation_single,
                                           batch_size=config.pipeline.dataloader.batch_size*2,
                                           shuffle=False)

    if args.auto_resume:
        # we get the last checkpoint and resume from there
        resume_from_checkpoint, run_name = get_last_checkpoint(config.pipeline.save_dir)
        if run_name is not None:
            if run_name[-1].isdigit():
                run_name = run_name[:-1] + str(int(run_name[-1]) + 1)
            else:
                run_name = run_name + "-PT2"
            # we name the run as the last one and append PT-X
            save_dir = os.path.join(config.pipeline.save_dir, run_name)
        else:
            resume_from_checkpoint = config.pipeline.lightning.resume_checkpoint
            run_name = get_run_name(config)
            save_dir = os.path.join(config.pipeline.save_dir, run_name)

    else:
        resume_from_checkpoint = config.pipeline.lightning.resume_checkpoint
        run_name = get_run_name(config)
        save_dir = os.path.join(config.pipeline.save_dir, run_name)

    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=run_name,
                               offline=config.pipeline.wandb.offline)

    loggers = [wandb_logger]

    checkpoint_callback = [ModelCheckpoint(dirpath=os.path.join(save_dir, 'checkpoints'), save_top_k=-1)]

    if len(config.pipeline.gpus) > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        strategy = 'ddp'
    else:
        strategy = None

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
                           save_dir=save_dir)

    trainer = Trainer(max_epochs=config.pipeline.epochs,
                      gpus=config.pipeline.gpus,
                      strategy=strategy,
                      default_root_dir=config.pipeline.save_dir,
                      precision=config.pipeline.precision,
                      logger=loggers,
                      check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                      val_check_interval=config.pipeline.lightning.val_check_interval,
                      num_sanity_val_steps=config.pipeline.lightning.num_sanity_val_steps,
                      resume_from_checkpoint=resume_from_checkpoint,
                      callbacks=checkpoint_callback,
                      log_every_n_steps=50)

    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True

    train(config)
