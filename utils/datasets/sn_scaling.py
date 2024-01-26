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


class SingleSNSourceDataset(Dataset):
    def __init__(self,
                 source_dataset: list,
                 scaling_list: np.array):
        r"""
            :param source_datasets: list of source datasets
        """
        super().__init__()

        self.source_dataset = source_dataset

        self.scaling_list = scaling_list

        self.source_len = len(self.source_dataset)

        self.ignore_label = self.source_dataset.ignore_label
        self.class2names = self.source_dataset.class2names
        self.voxel_size = self.source_dataset.voxel_size

    def __getitem__(self, i):
        source_data = self.source_dataset.__getitem__(i)

        source_coordinates = source_data['coordinates'] * self.voxel_size
        source_xyz = source_data['xyz']
        source_features = source_data['features']
        source_sem_labels = source_data['sem_labels']
        source_sampled_idx = source_data['sampled_idx']
        source_idx = source_data['idx']
        source_inverse_map = source_data['inverse_map']
        if len(self.scaling_list) > 1:
            scale_idx = np.arange(len(self.scaling_list))
            scale_idx = np.random.choice(scale_idx)
            scaling = self.scaling_list[scale_idx][0]
        else:
            scaling = self.scaling_list[0][0]

        source_coordinates[:, 0] = source_coordinates[:, 0] * scaling[0]
        source_coordinates[:, 1] = source_coordinates[:, 1] * scaling[1]
        source_coordinates[:, 2] = source_coordinates[:, 2] * scaling[2]

        quantized_coords, _, _, voxel_idx = ME.utils.sparse_quantize(source_coordinates.numpy(),
                                                              source_features.numpy(),
                                                              labels=source_sem_labels.numpy(),
                                                              ignore_label=self.ignore_label,
                                                              quantization_size=self.voxel_size,
                                                              return_index=True)
        source_features = source_features[voxel_idx]
        source_sem_labels = source_sem_labels[voxel_idx]

        return dict(coordinates=torch.from_numpy(quantized_coords),
                    xyz=source_xyz,
                    features=source_features,
                    sem_labels=source_sem_labels,
                    idx=source_idx,
                    sampled_idx=source_sampled_idx)

    def __len__(self):
        return self.source_len


class MultiSNSourceDataset(Dataset):
    def __init__(self,
                 source_datasets: list,
                 scaling_list: list):
        r"""
            :param source_datasets: list of source datasets
        """
        super().__init__()

        self.source_dataset0 = source_datasets[0]
        self.source_dataset1 = source_datasets[1]

        self.scaling_list0 = scaling_list[0]
        self.scaling_list1 = scaling_list[1]

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

        scale_idx0 = np.arange(self.scaling_list0.shape[0])
        scale_idx1 = np.arange(self.scaling_list1.shape[0])

        scale_idx0 = np.random.choice(scale_idx0)
        scale_idx1 = np.random.choice(scale_idx1)

        scaling0 = self.scaling_list0[scale_idx0]
        scaling1 = self.scaling_list1[scale_idx1]

        source_coordinates0[:, 0] = source_coordinates0[:, 0] * scaling0[0]
        source_coordinates0[:, 1] = source_coordinates0[:, 1] * scaling0[1]
        source_coordinates0[:, 2] = source_coordinates0[:, 2] * scaling0[2]

        source_coordinates1[:, 0] = source_coordinates1[:, 0] * scaling1[0]
        source_coordinates1[:, 1] = source_coordinates1[:, 1] * scaling1[1]
        source_coordinates1[:, 2] = source_coordinates1[:, 2] * scaling1[2]

        quantized_coords0, _, _, voxel_idx0 = ME.utils.sparse_quantize(source_coordinates0.numpy(),
                                                      source_features0.numpy(),
                                                      labels=source_sem_labels0.numpy(),
                                                      ignore_label=self.ignore_label,
                                                      quantization_size=self.voxel_size,
                                                      return_index=True)

        quantized_coords1, _, _, voxel_idx1 = ME.utils.sparse_quantize(source_coordinates1.numpy(),
                                                      source_features1.numpy(),
                                                      labels=source_sem_labels1.numpy(),
                                                      ignore_label=self.ignore_label,
                                                      quantization_size=self.voxel_size,
                                                      return_index=True)

        source_features0 = source_features0[voxel_idx0]
        source_sem_labels0 = source_sem_labels0[voxel_idx0]

        source_features1 = source_features1[voxel_idx1]
        source_sem_labels1 = source_sem_labels1[voxel_idx1]

        merged = dict(source_coordinates0=torch.from_numpy(quantized_coords0),
                      source_coordinates1=torch.from_numpy(quantized_coords1),
                      source_xyz0=source_xyz0,
                      source_xyz1=source_xyz1,
                      source_features0=source_features0,
                      source_features1=source_features1,
                      source_sem_labels0=source_sem_labels0,
                      source_sem_labels1=source_sem_labels1,
                      source_sampled_idx0=source_sampled_idx0,
                      source_sampled_idx1=source_sampled_idx1,
                      source_idx0=source_idx0,
                      source_idx1=source_idx1,
                      source_inverse_map0=source_inverse_map0,
                      source_inverse_map1=source_inverse_map1)
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
