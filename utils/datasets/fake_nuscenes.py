import os
import torch
import yaml
import numpy as np
import tqdm
from nuscenes.utils.splits import create_splits_scenes
from torchvision.transforms import Compose

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class FakeNuScenesDataset(BaseDataset):
    def __init__(self,
                 version: str = 'full',
                 phase: str = 'train',
                 dataset_path: str = '/data/csaltori/nuScenes-fake/data/',
                 mapping_path: str = '_resources/semantic-kitti.yaml',
                 weights_path: str = None,
                 voxel_size: float = 0.05,
                 use_intensity: bool = False,
                 augmentations: Compose = None,
                 sub_p: float = 1.0,
                 device: str = None,
                 num_classes: int = 7,
                 ignore_label: int = None,
                 use_cache: bool = True,
                 in_radius: float = 50.0,
                 is_test: bool = False):

        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path+'sequences/',
                         voxel_size=voxel_size,
                         sub_p=sub_p,
                         use_intensity=use_intensity,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label)

        splits = create_splits_scenes()

        if self.version == 'full':
            self.split = {'train': splits['train'],
                          'validation': splits['val']}
        else:
            self.split = {'train': splits['mini_train'],
                          'validation': splits['mini_val']}

        self.name = 'FakeNuScenesDataset'
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        self.pcd_path = []
        self.label_path = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        learning_map = -np.ones((max_key + 100), dtype=np.int32)
        learning_map[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.learning_map = learning_map

        for sequence in self.split[self.phase]:
            num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))

            for f in np.arange(num_frames):
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(f):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(f):06d}.label')
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)

        self.class2names = np.asarray(list(self.maps['mapped_labels'].values()))
        self.color_map = np.asarray(list(self.maps['mapped_color_map'].values()))/255.

        self.use_cache = use_cache
        self.CACHE = {}

        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
            if os.path.isfile(weights_path):
                stats = np.load(weights_path, allow_pickle=True)
                weights = stats.item().get('weights')
            else:
                weights = self.get_dataset_stats()
                main_path, _ = os.path.split(weights_path)
                os.makedirs(main_path, exist_ok=True)
                np.save(weights_path, {'weights': weights})

            self.sem_weights = weights

        self.in_R = in_radius
        self.augmentations = augmentations
        self.is_test = is_test

    def __len__(self):
        return len(self.pcd_path)

    def __getitem__(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        if i not in self.CACHE.keys():

            pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
            sem_labels = self.load_label_nusc(label_tmp)
            points = pcd[:, :3]

            mask = np.sum(np.square(points), axis=1) < self.in_R ** 2
            points = points[mask]
            sem_labels = sem_labels[mask]

            if self.use_intensity:
                colors = points[:, 3][..., np.newaxis]
            else:
                colors = np.ones((points.shape[0], 1), dtype=np.float32)

            data = {'points': points,
                    'colors': colors,
                    'sem_labels': sem_labels}

            if self.use_cache:
                self.CACHE[i] = data
        else:
            data = self.CACHE[i]

        points = data['points']
        colors = data['colors']
        sem_labels = data['sem_labels']

        sampled_idx = np.arange(points.shape[0])

        if self.phase == 'train' and self.augmentations is not None:
            sampled_idx = self.random_sample(points)
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            sem_labels = sem_labels[sampled_idx]
            points = self.augmentations(points)

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, _, _, voxel_idx, inverse_map = ME.utils.sparse_quantize(points,
                                                                  colors,
                                                                  labels=sem_labels,
                                                                  ignore_label=vox_ign_label,
                                                                  quantization_size=self.voxel_size,
                                                                  return_index=True,
                                                                  return_inverse=True)

        original_coords = points[voxel_idx]
        feats = colors[voxel_idx]
        sem_labels = sem_labels[voxel_idx]

        if isinstance(original_coords, np.ndarray):
            original_coords = torch.from_numpy(original_coords)

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(sem_labels, np.ndarray):
            sem_labels = torch.from_numpy(sem_labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        if isinstance(inverse_map, np.ndarray):
            inverse_map = torch.from_numpy(inverse_map)

        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None

        return {"coordinates": quantized_coords,
                "xyz": original_coords,
                "features": feats,
                "sem_labels": sem_labels,
                "sampled_idx": sampled_idx,
                "idx": torch.tensor(i),
                "inverse_map": inverse_map}

    def load_label_nusc(self, label_path: str):
        # frame_labels = np.fromfile(label_path, dtype=np.int32)
        # # sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
        # ins_labels = frame_labels >> 16
        # ins_labels = ins_labels.astype(np.int32)
        # sem_labels = (ins_labels // 1000).astype(np.uint8)
        # sem_labels = self.learning_map[sem_labels].astype(np.int32)
        sem_labels = np.fromfile(label_path, np.int32)  # [num_points]
        sem_labels = self.learning_map[sem_labels].astype(np.int32)

        return sem_labels

    def get_dataset_stats(self):
        weights = np.zeros(self.learning_map.max()+1)

        for l in tqdm.tqdm(range(len(self.label_path)), desc='Loading weights', leave=True):
            label_tmp = self.label_path[l]
            sem_labels = self.load_label_nusc(label_tmp)
            lbl, count = np.unique(sem_labels, return_counts=True)
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]

            weights[lbl] += count

        return weights

    def color_sem_labels(self, labels):
        # bgr -> rgb
        if labels.min() == -1:
            return self.color_map[labels+1]
        else:
            return self.color_map[labels]
