import os
import pickle
import numpy as np


from torch.utils.data import Dataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class BaseDataset(Dataset):
    def __init__(self,
                 version: str,
                 phase: str,
                 dataset_path: str,
                 voxel_size: float = 0.05,
                 sub_p: float = 1.0,
                 use_intensity: bool = False,
                 num_classes: int = 7,
                 ignore_label: int = None,
                 device: str = None):

        self.CACHE = {}
        self.version = version
        self.phase = phase
        self.dataset_path = dataset_path
        self.voxel_size = voxel_size  # in meter
        self.sub_p = sub_p
        self.use_intensity = use_intensity
        self.num_classes = num_classes

        self.ignore_label = ignore_label

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        self.device = device

        self.split = {'train': [],
                      'validation': []}

        self.maps = None
        self.color_map = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i: int):
        raise NotImplementedError

    def random_sample(self, points: np.ndarray, center: np.array = None) -> np.array:
        """
        :param points: input points of shape [N, 3]
        :param center: center to sample around, default is None, not used for now
        :return: np.ndarray of N' points sampled from input points
        """

        num_points = points.shape[0]

        if self.sub_p is not None:
            sampled_idx = np.random.choice(np.arange(num_points), int(self.sub_p * num_points), replace=False)
        else:
            sampled_idx = np.arange(num_points)

        return sampled_idx


class MultiSourceDataset(Dataset):
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

        self.ignore_label = self.source_dataset0.ignore_label
        self.class2names = self.source_dataset0.class2names

    @staticmethod
    def merge_data(source_data0, source_data1):
        # data arrive in a dict with keys [coordinates, xyz, features, sem_labels, sampled_idx, idx, inverse_map]

        merged = dict(source_coordinates0=source_data0['coordinates'],
                      source_coordinates1=source_data1['coordinates'],
                      source_xyz0=source_data0['xyz'],
                      source_xyz1=source_data1['xyz'],
                      source_features0=source_data0['features'],
                      source_features1=source_data1['features'],
                      source_sem_labels0=source_data0['sem_labels'],
                      source_sem_labels1=source_data1['sem_labels'],
                      source_sampled_idx0=source_data0['sampled_idx'],
                      source_sampled_idx1=source_data1['sampled_idx'],
                      source_idx0=source_data0['idx'],
                      source_idx1=source_data1['idx'],
                      source_inverse_map0=source_data0['inverse_map'],
                      source_inverse_map1=source_data1['inverse_map'])
        return merged

    def __getitem__(self, i):
        if i < self.source_len0:
            source_data0 = self.source_dataset0.__getitem__(i)
        else:
            # if required index is higher than len, we random select
            i_tmp = np.random.randint(0, self.source_len0)
            source_data0 = self.source_dataset0.__getitem__(i_tmp)

        if i < self.source_len1:
            source_data1 = self.source_dataset1.__getitem__(i)
        else:
            # if required index is higher than len, we random select
            i_tmp = np.random.randint(0, self.source_len1)
            source_data1 = self.source_dataset1.__getitem__(i_tmp)

        return self.merge_data(source_data0, source_data1)

    def __len__(self):
        return max(self.source_len0, self.source_len1)
