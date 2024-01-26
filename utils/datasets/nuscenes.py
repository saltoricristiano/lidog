import os
import torch
import yaml
import numpy as np
import tqdm
from torchvision.transforms import Compose

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class NuScenesDataset(BaseDataset):
    def __init__(self,
                 nusc,
                 version: str = 'full',
                 phase: str = 'train',
                 dataset_path: str = None,
                 mapping_path: str = '_resources/nuscenes2common.yaml',
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

        self.version = version

        self.nusc = nusc

        self.is_test = is_test
        self.name = 'NuScenesDataset'

        splits = create_splits_scenes()
        if self.version == 'v1.0-trainval':
            self.split = {'train': splits['train'],
                          'validation': splits['val']}
        else:
            self.split = {'train': splits['mini_train'],
                          'validation': splits['mini_val']}

        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        self.pcd_path = []
        self.label_path = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        learning_map = -np.ones((max_key + 100), dtype=np.int32)
        learning_map[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.learning_map = learning_map

        scenes_tokens = {}
        scenes = self.nusc.scene

        for s in scenes:
            # list tokens of the scenes
            scenes_tokens[s['name']] = s['token']

        for scene in self.split[self.phase]:

            # get scene token and scene
            scene_token = scenes_tokens[scene]
            scene_temp = self.nusc.get('scene', scene_token)

            # get first sample(frame) token
            sample_token = scene_temp['first_sample_token']

            # iterate over samples given tokens
            while sample_token != '':

                # get sample record
                sample_record = self.nusc.get('sample', sample_token)
                # get sensor token for given sample record
                sample_data_token = sample_record['data']['LIDAR_TOP']

                # get sample data of the lidar sensor
                lidar_data = self.nusc.get('sample_data', sample_data_token)

                # get lidar path
                pcd_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
                label_path = os.path.join(self.nusc.dataroot, self.nusc.get("lidarseg", sample_data_token)["filename"])

                assert os.path.exists(label_path) and os.path.exists(pcd_path)

                # update sample token with the next
                sample_token = sample_record['next']

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

        # we init colormap for instances
        max_inst_id = 100000
        self.inst_color_lut = np.random.uniform(low=0.0,
                                                high=1.0,
                                                size=(max_inst_id, 3))
        # force zero to a gray-ish color
        self.inst_color_lut[0] = np.full((3), 0.1)

        self.in_R = in_radius
        self.augmentations = augmentations

    def __len__(self):
        return len(self.pcd_path)

    def __getitem__(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        if i not in self.CACHE.keys():

            pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 5))
            sem_labels = self.load_label_nusc(label_tmp)

            assert pcd.shape[0] == sem_labels.shape[0], f'Points and labels have shape {pcd.shape[0]} and {sem_labels.shape[0]}'

            points = pcd[:, :3]

            if not self.is_test:
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
        sem_labels = np.fromfile(label_path, np.uint8)  # [num_points]
        sem_labels = self.learning_map[sem_labels].astype(np.int32)

        return sem_labels

    def get_dataset_stats(self):
        weights = np.zeros(self.learning_map.max()+1)
        max_instance = 0

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
