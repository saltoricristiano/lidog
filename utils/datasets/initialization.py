import os
from nuscenes import NuScenes
from configs import get_config
from utils.datasets.dataset import BaseDataset
from utils.datasets.semantickitti import SemanticKITTIDataset
from utils.datasets.nuscenes import NuScenesDataset
from utils.datasets.synth4d import Synth4DDataset
from utils.datasets.fake_kitti import FakeKITTIDataset
from utils.datasets.fake_nuscenes import FakeNuScenesDataset
from utils.datasets.fake_synth4d import FakeSynth4DDataset
from utils.datasets.synth4d_bev import Synth4DBEVDataset
from utils.datasets.semantickitti_bev import SemanticKITTIBEVDataset
from utils.datasets.nuscenes_bev import NuScenesBEVDataset
from utils.common.augmentation import get_augmentations

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATHS = get_config('configs/common/dataset_paths.yaml')
DATASETS_PATHS = {}

for name, path, m_path, w_path in list(zip(DATA_PATHS.datasets.name, DATA_PATHS.datasets.data_path, DATA_PATHS.datasets.mapping_path, DATA_PATHS.datasets.weights_path)):
    DATASETS_PATHS[name] = {'data_path': path, 'mapping_path': m_path, 'weights_path': w_path}


def get_dataset(dataset_name: str,
                voxel_size: float = 0.02,
                sub_p: float = 1.0,
                version: str = 'full',
                num_classes: int = 7,
                ignore_label: int = -1,
                use_cache: bool = False,
                augmentation_list: list = None,
                scale_bev: bool = False,
                decoder_2d_levels: list = ['block8'],
                bev_img_sizes: list = [240],
                bound_2d: float = 50.,
                method: str = None) -> (BaseDataset, BaseDataset):

    '''
        :param dataset_name: name of the dataset
        :param voxel_size: voxel size for voxelization
        :param sub_p: percentage of points sampled
        :param version: mini/full dataset
        :param num_classes: number of classes considered
        :param ignore_label: label to ignore
        :param use_cache: use or not caching during training
        :param augmentation_list: list of augmentations
        :param scale_bev: scale to img_size or not
        :param decoder_2d_levels: levels where 2D decoder is applied
        :param bev_img_sizes: dimension of the BEV map
        :param bound_2d: the 2D bounds for the 2D bev image
        :param method: method used for the completion task
        :return:
    '''

    dataset_path = DATASETS_PATHS[dataset_name]['data_path']
    mapping_path = DATASETS_PATHS[dataset_name]['mapping_path']
    weights_path = DATASETS_PATHS[dataset_name]['weights_path']

    if dataset_name == 'SemanticKITTI':

        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list)

        if method is None:

            training_dataset = SemanticKITTIDataset(dataset_path=dataset_path,
                                                    mapping_path=mapping_path,
                                                    version=version,
                                                    phase='train',
                                                    voxel_size=voxel_size,
                                                    sub_p=sub_p,
                                                    num_classes=num_classes,
                                                    ignore_label=ignore_label,
                                                    weights_path=weights_path,
                                                    use_cache=use_cache,
                                                    augmentations=augmentation_list)

            validation_dataset = SemanticKITTIDataset(dataset_path=dataset_path,
                                                      mapping_path=mapping_path,
                                                      version=version,
                                                      phase='validation',
                                                      voxel_size=voxel_size,
                                                      num_classes=num_classes,
                                                      ignore_label=ignore_label,
                                                      weights_path=weights_path,
                                                      use_cache=use_cache,
                                                      augmentations=None)
        else:
            raise NotImplementedError

    elif dataset_name == 'nuScenes':

        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list)

        version = 'v1.0-trainval' if version == 'full' else 'v1.0-mini'
        nusc = NuScenes(version=version, dataroot=dataset_path, verbose=True)

        training_dataset = NuScenesDataset(nusc=nusc,
                                           dataset_path=dataset_path,
                                           mapping_path=mapping_path,
                                           version=version,
                                           phase='train',
                                           voxel_size=voxel_size,
                                           sub_p=sub_p,
                                           num_classes=num_classes,
                                           ignore_label=ignore_label,
                                           weights_path=weights_path,
                                           use_cache=use_cache,
                                           augmentations=augmentation_list)

        validation_dataset = NuScenesDataset(nusc=nusc,
                                             dataset_path=dataset_path,
                                             mapping_path=mapping_path,
                                             version=version,
                                             phase='validation',
                                             voxel_size=voxel_size,
                                             num_classes=num_classes,
                                             ignore_label=ignore_label,
                                             weights_path=weights_path,
                                             use_cache=use_cache,
                                             augmentations=None)

    elif dataset_name == 'Synth4D-kitti':
        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list)

        training_split_path = os.path.join(ABSOLUTE_PATH, '_split/kitti_synth/training_split.pkl')
        validation_split_path = os.path.join(ABSOLUTE_PATH, '_split/kitti_synth/validation_split.pkl')

        if method is None:

            training_dataset = Synth4DDataset(dataset_path=dataset_path,
                                              mapping_path=mapping_path,
                                              version=version,
                                              phase='train',
                                              voxel_size=voxel_size,
                                              sub_p=sub_p,
                                              num_classes=num_classes,
                                              ignore_label=ignore_label,
                                              weights_path=weights_path,
                                              use_cache=use_cache,
                                              augmentations=augmentation_list,
                                              sensor='hdl64e',
                                              split_path=training_split_path)

            validation_dataset = Synth4DDataset(dataset_path=dataset_path,
                                                mapping_path=mapping_path,
                                                version=version,
                                                phase='validation',
                                                voxel_size=voxel_size,
                                                num_classes=num_classes,
                                                ignore_label=ignore_label,
                                                weights_path=weights_path,
                                                use_cache=use_cache,
                                                augmentations=None,
                                                sensor='hdl32e',
                                                split_path=validation_split_path)

        else:
            raise ValueError('Unknown method: {}'.format(method))

    elif dataset_name == 'Synth4D-nuscenes':

        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list)

        training_split_path = os.path.join(ABSOLUTE_PATH, '_split/nuscenes_synth/training_split.pkl')
        validation_split_path = os.path.join(ABSOLUTE_PATH, '_split/nuscenes_synth/validation_split.pkl')

        if method is None:
            training_dataset = Synth4DDataset(dataset_path=dataset_path,
                                              mapping_path=mapping_path,
                                              version=version,
                                              phase='train',
                                              voxel_size=voxel_size,
                                              sub_p=sub_p,
                                              num_classes=num_classes,
                                              ignore_label=ignore_label,
                                              weights_path=weights_path,
                                              use_cache=use_cache,
                                              augmentations=augmentation_list,
                                              sensor='hdl32e',
                                              split_path=training_split_path)

            validation_dataset = Synth4DDataset(dataset_path=dataset_path,
                                                mapping_path=mapping_path,
                                                version=version,
                                                phase='validation',
                                                voxel_size=voxel_size,
                                                num_classes=num_classes,
                                                ignore_label=ignore_label,
                                                weights_path=weights_path,
                                                use_cache=use_cache,
                                                augmentations=None,
                                                sensor='hdl32e',
                                                split_path=validation_split_path)
        else:
            raise ValueError('Unknown method: {}'.format(method))

    elif dataset_name == 'FakeSynth4D-kitti':
        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list)

        training_split_path = os.path.join(ABSOLUTE_PATH, '_split/kitti_synth/training_split.pkl')
        validation_split_path = os.path.join(ABSOLUTE_PATH, '_split/kitti_synth/validation_split.pkl')

        training_dataset = FakeSynth4DDataset(dataset_path=dataset_path,
                                              mapping_path=mapping_path,
                                              version=version,
                                              phase='train',
                                              voxel_size=voxel_size,
                                              sub_p=sub_p,
                                              num_classes=num_classes,
                                              ignore_label=ignore_label,
                                              weights_path=weights_path,
                                              use_cache=use_cache,
                                              augmentations=augmentation_list,
                                              sensor='hdl64e',
                                              split_path=training_split_path)

        validation_dataset = FakeSynth4DDataset(dataset_path=dataset_path,
                                                mapping_path=mapping_path,
                                                version=version,
                                                phase='validation',
                                                voxel_size=voxel_size,
                                                num_classes=num_classes,
                                                ignore_label=ignore_label,
                                                weights_path=weights_path,
                                                use_cache=use_cache,
                                                augmentations=None,
                                                sensor='hdl64e',
                                                split_path=validation_split_path)

    elif dataset_name == 'FakeSynth4D-nuscenes':

        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list)

        training_split_path = os.path.join(ABSOLUTE_PATH, '_split/nuscenes_synth/training_split.pkl')
        validation_split_path = os.path.join(ABSOLUTE_PATH, '_split/nuscenes_synth/validation_split.pkl')

        training_dataset = FakeSynth4DDataset(dataset_path=dataset_path,
                                              mapping_path=mapping_path,
                                              version=version,
                                              phase='train',
                                              voxel_size=voxel_size,
                                              sub_p=sub_p,
                                              num_classes=num_classes,
                                              ignore_label=ignore_label,
                                              weights_path=weights_path,
                                              use_cache=use_cache,
                                              augmentations=augmentation_list,
                                              sensor='hdl32e',
                                              split_path=training_split_path)

        validation_dataset = FakeSynth4DDataset(dataset_path=dataset_path,
                                                mapping_path=mapping_path,
                                                version=version,
                                                phase='validation',
                                                voxel_size=voxel_size,
                                                num_classes=num_classes,
                                                ignore_label=ignore_label,
                                                weights_path=weights_path,
                                                use_cache=use_cache,
                                                augmentations=None,
                                                sensor='hdl32e',
                                                split_path=validation_split_path)

    elif dataset_name == 'FakeKITTI':

        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list)

        training_dataset = FakeKITTIDataset(dataset_path=dataset_path,
                                            mapping_path=mapping_path,
                                            version=version,
                                            phase='train',
                                            voxel_size=voxel_size,
                                            sub_p=sub_p,
                                            num_classes=num_classes,
                                            ignore_label=ignore_label,
                                            weights_path=weights_path,
                                            use_cache=use_cache,
                                            augmentations=augmentation_list)

        v_dataset_path = DATASETS_PATHS['SemanticKITTI']['data_path']
        v_mapping_path = DATASETS_PATHS['SemanticKITTI']['mapping_path']
        v_weights_path = DATASETS_PATHS['SemanticKITTI']['weights_path']

        validation_dataset = SemanticKITTIDataset(dataset_path=v_dataset_path,
                                                  mapping_path=v_mapping_path,
                                                  version=version,
                                                  phase='validation',
                                                  voxel_size=voxel_size,
                                                  num_classes=num_classes,
                                                  ignore_label=ignore_label,
                                                  weights_path=v_weights_path,
                                                  use_cache=use_cache,
                                                  augmentations=None)

    elif dataset_name == 'FakeNuScenes':

        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list)

        training_dataset = FakeNuScenesDataset(dataset_path=dataset_path,
                                               mapping_path=mapping_path,
                                               version=version,
                                               phase='train',
                                               voxel_size=voxel_size,
                                               sub_p=sub_p,
                                               num_classes=num_classes,
                                               ignore_label=ignore_label,
                                               weights_path=weights_path,
                                               use_cache=use_cache,
                                               augmentations=augmentation_list)

        v_dataset_path = DATASETS_PATHS['nuScenes']['data_path']
        v_mapping_path = DATASETS_PATHS['nuScenes']['mapping_path']
        v_weights_path = DATASETS_PATHS['nuScenes']['weights_path']

        version = 'v1.0-trainval' if version == 'full' else 'v1.0-mini'
        nusc = NuScenes(version=version, dataroot=v_dataset_path, verbose=True)

        validation_dataset = NuScenesDataset(nusc=nusc,
                                             dataset_path=v_dataset_path,
                                             mapping_path=v_mapping_path,
                                             version=version,
                                             phase='validation',
                                             voxel_size=voxel_size,
                                             num_classes=num_classes,
                                             ignore_label=ignore_label,
                                             weights_path=v_weights_path,
                                             use_cache=use_cache,
                                             augmentations=None)

    elif dataset_name == 'Synth4D-kitti-BEV':
        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list, is_bev=True)

        training_split_path = os.path.join(ABSOLUTE_PATH, '_split/kitti_synth/training_split.pkl')
        validation_split_path = os.path.join(ABSOLUTE_PATH, '_split/kitti_synth/validation_split.pkl')

        training_dataset = Synth4DBEVDataset(dataset_path=dataset_path,
                                             mapping_path=mapping_path,
                                             version=version,
                                             phase='train',
                                             voxel_size=voxel_size,
                                             sub_p=sub_p,
                                             num_classes=num_classes,
                                             ignore_label=ignore_label,
                                             weights_path=weights_path,
                                             use_cache=use_cache,
                                             augmentations=augmentation_list,
                                             sensor='hdl64e',
                                             split_path=training_split_path,
                                             scale_bev=scale_bev,
                                             decoder_2d_levels=decoder_2d_levels,
                                             bev_img_sizes=dict(zip(decoder_2d_levels, bev_img_sizes)),
                                             bound_2d=bound_2d)

        validation_dataset = Synth4DDataset(dataset_path=dataset_path,
                                            mapping_path=mapping_path,
                                            version=version,
                                            phase='validation',
                                            voxel_size=voxel_size,
                                            num_classes=num_classes,
                                            ignore_label=ignore_label,
                                            weights_path=weights_path,
                                            use_cache=use_cache,
                                            augmentations=None,
                                            sensor='hdl64e',
                                            split_path=validation_split_path)

    elif dataset_name == 'Synth4D-nuscenes-BEV':
        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list, is_bev=True)

        training_split_path = os.path.join(ABSOLUTE_PATH, '_split/nuscenes_synth/training_split.pkl')
        validation_split_path = os.path.join(ABSOLUTE_PATH, '_split/nuscenes_synth/validation_split.pkl')

        training_dataset = Synth4DBEVDataset(dataset_path=dataset_path,
                                             mapping_path=mapping_path,
                                             version=version,
                                             phase='train',
                                             voxel_size=voxel_size,
                                             sub_p=sub_p,
                                             num_classes=num_classes,
                                             ignore_label=ignore_label,
                                             weights_path=weights_path,
                                             use_cache=use_cache,
                                             augmentations=augmentation_list,
                                             sensor='hdl32e',
                                             split_path=training_split_path,
                                             scale_bev=scale_bev,
                                             decoder_2d_levels=decoder_2d_levels,
                                             bev_img_sizes=dict(zip(decoder_2d_levels, bev_img_sizes)),
                                             bound_2d=bound_2d)

        validation_dataset = Synth4DDataset(dataset_path=dataset_path,
                                            mapping_path=mapping_path,
                                            version=version,
                                            phase='validation',
                                            voxel_size=voxel_size,
                                            num_classes=num_classes,
                                            ignore_label=ignore_label,
                                            weights_path=weights_path,
                                            use_cache=use_cache,
                                            augmentations=None,
                                            sensor='hdl32e',
                                            split_path=validation_split_path)

    elif dataset_name == 'SemanticKITTI-BEV':
        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list, is_bev=True)

        training_dataset = SemanticKITTIBEVDataset(dataset_path=dataset_path,
                                                   mapping_path=mapping_path,
                                                   version=version,
                                                   phase='train',
                                                   voxel_size=voxel_size,
                                                   sub_p=sub_p,
                                                   num_classes=num_classes,
                                                   ignore_label=ignore_label,
                                                   weights_path=weights_path,
                                                   use_cache=use_cache,
                                                   augmentations=augmentation_list,
                                                   decoder_2d_levels=decoder_2d_levels,
                                                   bev_img_sizes=dict(zip(decoder_2d_levels, bev_img_sizes)),
                                                   bound_2d=bound_2d)

        validation_dataset = SemanticKITTIDataset(dataset_path=dataset_path,
                                                  mapping_path=mapping_path,
                                                  version=version,
                                                  phase='validation',
                                                  voxel_size=voxel_size,
                                                  num_classes=num_classes,
                                                  ignore_label=ignore_label,
                                                  weights_path=weights_path,
                                                  use_cache=use_cache,
                                                  augmentations=None)
    elif dataset_name == 'nuScenes-BEV':
        if augmentation_list is not None:
            augmentation_list = get_augmentations(augmentation_list, is_bev=True)

        version = 'v1.0-trainval' if version == 'full' else 'v1.0-mini'
        nusc = NuScenes(version=version, dataroot=dataset_path, verbose=True)

        training_dataset = NuScenesBEVDataset(nusc=nusc,
                                              dataset_path=dataset_path,
                                              mapping_path=mapping_path,
                                              version=version,
                                              phase='train',
                                              voxel_size=voxel_size,
                                              sub_p=sub_p,
                                              num_classes=num_classes,
                                              ignore_label=ignore_label,
                                              weights_path=weights_path,
                                              use_cache=use_cache,
                                              augmentations=augmentation_list,
                                              decoder_2d_levels=decoder_2d_levels,
                                              bev_img_sizes=dict(zip(decoder_2d_levels, bev_img_sizes)),
                                              bound_2d=bound_2d)

        validation_dataset = NuScenesDataset(nusc=nusc,
                                             dataset_path=dataset_path,
                                             mapping_path=mapping_path,
                                             version=version,
                                             phase='validation',
                                             voxel_size=voxel_size,
                                             num_classes=num_classes,
                                             ignore_label=ignore_label,
                                             weights_path=weights_path,
                                             use_cache=use_cache,
                                             augmentations=None)

    else:
        raise NotImplementedError

    return training_dataset, validation_dataset

