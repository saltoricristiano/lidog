import os
import torch
import yaml
import numpy as np
import tqdm
import pickle
from torchvision.transforms import Compose

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class FakeSynth4DDataset(BaseDataset):
    def __init__(self,
                 version: str = 'full',
                 phase: str = 'train',
                 dataset_path: str = '/data/csaltori/Synth4D/data/',
                 mapping_path: str = '_resources/semantic-kitti.yaml',
                 split_path: str = None,
                 sensor: str = 'hdl64e',
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
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_p=sub_p,
                         use_intensity=use_intensity,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label)

        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        split = 'training_split' if phase == 'train' else 'validation_split'

        self.split = load_obj(split_path)
        self.sensor = sensor

        if self.sensor == 'hdl64e':
            self.name = 'FakeSyntheticKITTIDataset'
        elif self.sensor == 'hdl32e':
            self.name = 'FakeSyntheticNuScenesDataset'
        else:
            raise NotImplementedError

        if self.version == 'mini':
            _split = {}
            for town in self.split.keys():
                _split[town] = np.random.choice(self.split[town], 100)
            self.split = _split

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = -np.ones((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.learning_map = remap_lut_val

        self.pcd_path = []
        self.label_path = []

        for town in self.split.keys():
            pc_path = os.path.join(self.dataset_path, town, 'velodyne')
            lbl_path = os.path.join(self.dataset_path, town, 'labels')
            self.pcd_path.extend([os.path.join(pc_path, str(f).zfill(6)+'.bin') for f in np.sort(self.split[town])])
            self.label_path.extend([os.path.join(lbl_path, str(f).zfill(6)+'.label') for f in np.sort(self.split[town])])

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
        self.use_cache = use_cache

        self.class2names = np.asarray(list(self.maps['mapped_labels'].values()))
        self.color_map = np.asarray(list(self.maps['mapped_color_map'].values()))/255.

    def __getitem__(self, i):
        pc_path = self.pcd_path[i]
        points = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))

        label_path = self.label_path[i]

        if i not in self.CACHE:

            if not os.path.exists(label_path):
                labels = np.zeros(np.shape(points)[0], dtype=np.int32)
            else:
                if self.name == 'FakeSyntheticKITTIDataset':
                    labels = np.fromfile(label_path, dtype=np.int32).reshape((-1))
                else:
                    labels = np.fromfile(label_path, dtype=np.int8).reshape((-1))

                labels = self.learning_map[labels]
            pcd = points[:, :3]

            if self.use_intensity:
                colors = points[:, 3][..., np.newaxis]
            else:
                colors = np.ones((points.shape[0], 1), dtype=np.float32)

            data = {'points': pcd, 'colors': colors, 'labels': labels}

            if self.use_cache:
                self.CACHE[i] = data
        else:
            data = self.CACHE[i]

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        sampled_idx = np.arange(points.shape[0])

        if self.phase == 'train' and self.augmentations is not None:
            sampled_idx = self.random_sample(points)

            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            if self.augmentations is not None:
                points = self.augmentations(points)

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, _, _, voxel_idx, inverse_map = ME.utils.sparse_quantize(points,
                                                                                  colors,
                                                                                  labels=labels,
                                                                                  ignore_label=vox_ign_label,
                                                                                  quantization_size=self.voxel_size,
                                                                                  return_index=True,
                                                                                  return_inverse=True)

        original_coords = points[voxel_idx]
        feats = colors[voxel_idx]
        sem_labels = labels[voxel_idx]

        if isinstance(points, np.ndarray):
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

    def __len__(self):
        return len(self.pcd_path)

    def get_dataset_stats(self):
        weights = np.zeros(self.learning_map.max()+1)

        for l in tqdm.tqdm(range(len(self.pcd_path)), desc='Loading weights', leave=True):
            pc_path = self.pcd_path[l]
            label_tmp = self.label_path[l]
            labels = np.fromfile(label_tmp, dtype=np.int32).reshape((-1))
            sem_labels = self.learning_map[labels]
            lbl, count = np.unique(sem_labels, return_counts=True)
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]

            weights[lbl] += count

        return weights
