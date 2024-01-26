import os
import pickle
import numpy as np
import torch

import MinkowskiEngine as ME
from torch.utils.data import Dataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class PointCutMixSourceDataset(Dataset):
    def __init__(self,
                 source_datasets: list):
        r"""
            :param source_datasets: list of source datasets
        """
        super().__init__()

        self.source_dataset0 = source_datasets[0]
        self.source_dataset1 = source_datasets[1]

        self.number_source = len(source_datasets)

        if self.number_source > 2:
            raise NotImplementedError

        self.source_len0 = len(self.source_dataset0)
        self.source_len1 = len(self.source_dataset1)

        self.source1_idx = np.arange(0, self.source_len1)
        np.random.shuffle(self.source1_idx)

        self.ignore_label = self.source_dataset0.ignore_label
        self.class2names = self.source_dataset0.class2names
        self.voxel_size = self.source_dataset0.voxel_size

    def merge_data(self, source_data0, source_data1):
        # data arrive in a dict with keys [coordinates, xyz, features, sem_labels, sampled_idx, idx, inverse_map]
        source_coordinates0 = source_data0['coordinates'] * self.voxel_size
        source_coordinates1 = source_data1['coordinates'] * self.voxel_size
        source_xyz0 = source_data0['xyz']
        source_xyz1 = source_data1['xyz']
        source_features0 = source_data0['features']
        source_features1 = source_data1['features']
        source_sem_labels0 = source_data0['sem_labels']
        source_sem_labels1 = source_data1['sem_labels']
        source_sampled_idx0 = source_data0['sampled_idx']
        source_sampled_idx1 = source_data1['sampled_idx']
        source_idx0 = source_data0['idx']
        source_idx1 = source_data1['idx']
        source_inverse_map0 = source_data0['inverse_map']
        source_inverse_map1 = source_data1['inverse_map']

        coordinates = [source_coordinates0, source_coordinates1]
        xyz = [source_xyz0, source_xyz1]
        features = [source_features0, source_features1]
        sem_labels = [source_sem_labels0, source_sem_labels1]
        sampled_idx = [source_sampled_idx0, source_sampled_idx1]

        selected_source = np.random.choice([0, 1])
        selected_target = (selected_source+1) % 2

        source_coordinates = coordinates[selected_source]
        source_xyz = xyz[selected_source]
        source_features = features[selected_source]
        source_sem_labels = sem_labels[selected_source]
        source_sampled_idx = sampled_idx[selected_source]

        target_coordinates = coordinates[selected_target]
        target_xyz = xyz[selected_target]
        target_features = features[selected_target]
        target_sem_labels = sem_labels[selected_target]
        target_sampled_idx = sampled_idx[selected_target]

        idx = torch.cat([source_idx0.view(1, -1), source_idx1.view(1, -1)], dim=0)

        _, _, _, _, inverse_map = ME.utils.sparse_quantize(source_coordinates.numpy(),
                                                      source_features.numpy(),
                                                      labels=source_sem_labels.numpy(),
                                                      ignore_label=self.ignore_label,
                                                      quantization_size=10.0,
                                                      return_index=True,
                                                      return_inverse=True)

        vox_idx, count = np.unique(inverse_map, return_counts=True)
        selected_voxels = np.random.choice(vox_idx[count > 300], 4, replace=False)

        new_mixed_coords = [target_coordinates]
        new_mixed_feats = [target_features]
        new_mixed_xyz = [target_xyz]
        new_mixed_labels = [target_sem_labels]
        new_mixed_sampled_idx = [target_sampled_idx]

        for sv in selected_voxels:
            cut_idx = inverse_map == sv
            new_mixed_coords.append(source_coordinates[cut_idx])
            new_mixed_feats.append(source_features[cut_idx])
            new_mixed_xyz.append(source_xyz[cut_idx])
            new_mixed_labels.append(source_sem_labels[cut_idx])
            new_mixed_sampled_idx.append(source_sampled_idx[cut_idx])

        new_mixed_coords = torch.cat(new_mixed_coords, dim=0)
        new_mixed_feats = torch.cat(new_mixed_feats, dim=0)
        new_mixed_xyz = torch.cat(new_mixed_xyz, dim=0)
        new_mixed_labels = torch.cat(new_mixed_labels, dim=0)
        new_mixed_sampled_idx = torch.cat(new_mixed_sampled_idx, dim=0)

        quantized_coords, _, _, voxel_idx = ME.utils.sparse_quantize(new_mixed_coords.numpy(),
                                                      new_mixed_feats.numpy(),
                                                      labels=new_mixed_labels.numpy(),
                                                      ignore_label=self.ignore_label,
                                                      quantization_size=self.voxel_size,
                                                      return_index=True)

        new_mixed_feats = new_mixed_feats[voxel_idx]
        new_mixed_xyz = new_mixed_xyz[voxel_idx]
        new_mixed_labels = new_mixed_labels[voxel_idx]
        new_mixed_sampled_idx = new_mixed_sampled_idx[voxel_idx]

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        merged = dict(coordinates=quantized_coords,
                      xyz=new_mixed_xyz,
                      features=new_mixed_feats,
                      sem_labels=new_mixed_labels,
                      idx=idx,
                      sampled_idx=new_mixed_sampled_idx)
        return merged

    def __getitem__(self, i):
        if i < self.source_len0:
            source_data0 = self.source_dataset0.__getitem__(i)
        else:
            # if required index is higher than len, we random select
            i_tmp = np.random.randint(0, self.source_len0)
            source_data0 = self.source_dataset0.__getitem__(i_tmp)

        if i < self.source_len1:
            j = self.source1_idx[i]
            source_data1 = self.source_dataset1.__getitem__(j)
        else:
            # if required index is higher than len, we random select
            i_tmp = np.random.randint(0, self.source_len1)
            source_data1 = self.source_dataset1.__getitem__(i_tmp)

        return self.merge_data(source_data0, source_data1)

    def __len__(self):
        return max(self.source_len0, self.source_len1)
