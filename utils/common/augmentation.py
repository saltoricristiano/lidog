import numpy as np
from scipy.linalg import expm, norm
from torchvision.transforms import Compose
from utils.common.transforms import ComposeBEV


class RandomRotation:

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, coords, r=None, is_bev=False):
        if r is None:
            R = self._M(
                np.random.rand(3) - 0.5, np.pi/4 * (np.random.rand(1) - 0.5))
        else:
            R = r
        if not is_bev:
            return coords @ R
        else:
            return coords @ R, R


class RandomScale:

    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords, s=None, is_bev=False):
        if s is None:
            s_x = self.scale * np.random.rand(1) + self.bias
            s_y = self.scale * np.random.rand(1) + self.bias
            s_z = self.scale * np.random.rand(1) + self.bias
        else:
            s_x, s_y, s_z = s

        coords[:, 0] = coords[:, 0] * s_x
        coords[:, 1] = coords[:, 1] * s_y
        coords[:, 2] = coords[:, 2] * s_z
        if not is_bev:
            return coords
        else:
            return coords, [s_x, s_y, s_z]


class RandomShear:

    def __call__(self, coords):
        T = np.eye(3) + np.random.randn(3, 3)
        return coords @ T


class RandomTranslation:

    def __call__(self, coords):
        trans = 0.05 * np.random.randn(1, 3)
        return coords + trans


def get_augmentations(augs: list, is_bev: bool = False):
    aug_list = []
    for a in augs:
        if a == 'RandomRotation':
            aug_list.append(RandomRotation())
        elif a == 'RandomScale':
            aug_list.append(RandomScale(0.9, 1.1))
        else:
            raise NotImplementedError
    if not is_bev:
        return Compose(aug_list)
    else:
        return ComposeBEV(aug_list)
