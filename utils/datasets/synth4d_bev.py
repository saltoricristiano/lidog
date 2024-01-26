import os
from typing import Union

import torch
import yaml
import numpy as np
import tqdm
import pickle
import math
from torchvision.transforms import Compose
from utils.common.transforms import ComposeBEV
from torch.utils.data import Dataset

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def sanity_check(values):
    nan_check = torch.isnan(values).sum() == 0
    sum_check = values.sum(dim=-1).float() == 1
    min_check = values.min() >= 0.
    max_check = values.max() <= 1.

    return nan_check and sum_check and min_check and max_check


class Synth4DBEVDataset(BaseDataset):
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
                 augmentations: [Compose, ComposeBEV] = None,
                 sub_p: float = 1.0,
                 device: str = None,
                 num_classes: int = 7,
                 ignore_label: int = None,
                 use_cache: bool = True,
                 in_radius: float = 50.0,
                 is_test: bool = False,
                 scale_bev: bool = False,
                 use_bounds: bool = True,
                 decoder_2d_levels: list = ['block8'],
                 bev_img_sizes: dict = {'bottle': 240, 'block6': 240, 'block7': 240, 'block8': 240},
                 bound_2d: float = 50.0,
                 scaling_factor: dict = {'bottle': 1.0, 'block6': 1.0, 'block7': 1.0, 'block8': 1.0},
                 augment_bev: bool = True):

        r'''
        :param version: full or mini
        :param phase: val or train
        :param dataset_path: path to the dataset
        :param mapping_path: path to the mapping file for labels
        :param split_path: path to the split file
        :param sensor: which sensor to use (hdl64e, hdl32e)
        :param weights_path: path to the weights for weighted losses
        :param voxel_size: voxel size for the sparse tensor
        :param use_intensity: if to use intensity or not (not used)
        :param augmentations: data augmentations
        :param sub_p: selection percentage as augmentation
        :param device: not used
        :param num_classes: number of output classes
        :param ignore_label: label to ignore
        :param use_cache: if to load or not data in the cache
        :param in_radius: range radius for the sparse tensor
        :param is_test: if the dataset is used for testing
        :param scale_bev: if to scale the bev images if scaling is in the augmentations
        :param use_bounds: always true, filters bev image bounds to ask to complete only the visible part for t
        :param decoder_2d_levels: level at which is inserted the decoder for the bev task
        :param bev_img_sizes: size of the bev image
        :param bound_2d: if to use bounds, this is the max depth for bev images
        :param scaling_factor: for the scaling experiments, downscale the bev images
        :param augment_bev: applies augmentation to the bev images
        '''

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
            self.name = 'SyntheticKITTIDataset'
            self.dataset_path = os.path.join(self.dataset_path, 'kitti_synth')

        elif self.sensor == 'hdl32e':
            self.name = 'SyntheticNuScenesDataset'
            self.dataset_path = os.path.join(self.dataset_path, 'nuscenes_synth')

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

        self.path_list = []

        for town in self.split.keys():
            pc_path = os.path.join(self.dataset_path, town, 'velodyne')
            self.path_list.extend([os.path.join(pc_path, str(f)+'.npy') for f in np.sort(self.split[town])])

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
        self.augment_bev = augment_bev
        if self.augmentations is not None:
            self.bev_augmentations = ComposeBEV([self.augmentations.transforms[0]])
        self.is_test = is_test
        self.use_cache = use_cache

        self.class2names = np.asarray(list(self.maps['mapped_labels'].values()))
        self.color_map = np.asarray(list(self.maps['mapped_color_map'].values()))/255.
        self.scale_bev = scale_bev
        self.grid_bounds = [[-60, 60], [-60, 60], [-10, 8]]
        self.grid_bounds2d = [[-bound_2d, bound_2d], [-bound_2d, bound_2d], [-10, 8]]
        self.use_bounds = use_bounds

        self.decoder_2d_levels = decoder_2d_levels
        self.scaling_factor = scaling_factor

        self.bev_img_sizes = bev_img_sizes

        self.bev_converters = {}

        for key in self.decoder_2d_levels:
            xsize = (self.grid_bounds2d[0][1] - self.grid_bounds2d[0][0]) / (self.bev_img_sizes[key])
            ysize = (self.grid_bounds2d[1][1] - self.grid_bounds2d[1][0]) / (self.bev_img_sizes[key])
            zsize = 0.3

            self.bev_converters[key] = PC2ImgConverter(imgChannel=1,
                                                       xRange=self.grid_bounds2d[0],
                                                       yRange=self.grid_bounds2d[1],
                                                       zRange=self.grid_bounds2d[2],
                                                       xGridSize=xsize,
                                                       yGridSize=ysize,
                                                       zGridSize=zsize)

    def filter_bounds(self, points):

        pts_x = points[:, 0]
        pts_y = points[:, 1]
        pts_z = points[:, 2]

        in_bound_x = np.logical_and(self.grid_bounds[0][0] < pts_x, pts_x < self.grid_bounds[0][1])
        in_bound_y = np.logical_and(self.grid_bounds[1][0] < pts_y, pts_y < self.grid_bounds[1][1])
        in_bound_z = np.logical_and(self.grid_bounds[2][0] < pts_z, pts_z < self.grid_bounds[2][1])
        in_bound_idx = np.logical_and(in_bound_x, np.logical_and(in_bound_y, in_bound_z))

        ego_x = np.logical_and(-3 < pts_x, pts_x < 3)
        ego_y = np.logical_and(-2 < pts_y, pts_y < 2)
        ego_idx = np.logical_not(np.logical_and(ego_x, ego_y))

        in_bound_idx = np.logical_and(in_bound_idx, ego_idx)

        return in_bound_idx

    @staticmethod
    def get_soft(t_vector, eps=0.25):

        max_val = 1 - eps
        min_val = eps / (t_vector.shape[-1] - 1)

        t_soft = torch.empty(t_vector.shape)
        t_soft[t_vector == 0] = min_val
        t_soft[t_vector == 1] = max_val

        return t_soft

    def __getitem__(self, i):

        pc_path = self.path_list[i]
        input_points = np.load(pc_path).astype(np.float32)

        dir, file = os.path.split(pc_path)
        dir, _ = os.path.split(dir)
        label_path = os.path.join(dir, 'labels', file[:-4] + '.npy')

        _, town = os.path.split(dir)

        if i not in self.CACHE:

            if not os.path.exists(label_path):
                labels = np.zeros(np.shape(input_points)[0], dtype=np.int32)
            else:
                labels = np.load(label_path).astype(np.int32).reshape([-1])
                labels = self.learning_map[labels]

            if self.use_intensity:
                colors = input_points[:, 3][..., np.newaxis]
            else:
                colors = np.ones((input_points.shape[0], 1), dtype=np.float32)

            data = {'points': input_points, 'colors': colors, 'labels': labels}

            if self.use_cache:
                self.CACHE[i] = data
        else:
            data = self.CACHE[i]

        input_points = data['points']
        points = input_points[:, :3]
        colors = data['colors']
        labels = data['labels']

        sampled_idx = np.arange(points.shape[0])

        if self.phase == 'train' and self.augmentations is not None:

            sampled_idx = self.random_sample(points)

            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]
            points, tr = self.augmentations(points)

        if self.use_bounds:
            in_bound_idx = self.filter_bounds(points)

            points = points[in_bound_idx]
            labels = labels[in_bound_idx]
            colors = colors[in_bound_idx]

            sampled_idx = sampled_idx[in_bound_idx]

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, _, quantized_labels, voxel_idx, inverse_map = ME.utils.sparse_quantize(points,
                                                                                  colors,
                                                                                  labels=labels,
                                                                                  ignore_label=vox_ign_label,
                                                                                  quantization_size=self.voxel_size,
                                                                                  return_index=True,
                                                                                  return_inverse=True)

        bev_points = (quantized_coords*self.voxel_size).astype(np.float32)

        bev_labels = {}
        bev_selected_idx = {}

        for key in self.decoder_2d_levels:
            bev_labels_lvl, bev_selected_idx_lvl = self.bev_converters[key].getBEVImageNew(bev_points, quantized_labels)

            bev_labels[key] = torch.from_numpy(bev_labels_lvl).long()
            bev_selected_idx[key] = torch.from_numpy(bev_selected_idx_lvl).long()

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
            sem_labels = torch.from_numpy(sem_labels).long()

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        if isinstance(inverse_map, np.ndarray):
            inverse_map = torch.from_numpy(inverse_map)

        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None

        return_dict = {"coordinates": quantized_coords,
                       "xyz": original_coords,
                       "features": feats,
                       "sem_labels": sem_labels,
                       "sampled_idx": sampled_idx,
                       "idx": torch.tensor(i),
                       "inverse_map": inverse_map,
                       "bev_labels": bev_labels,
                       'bev_selected_idx': bev_selected_idx}

        return return_dict

    @staticmethod
    def globalize(pts_temp, trans):
        points = np.ones((pts_temp.shape[0], 4))
        points[:, 0:3] = pts_temp[:, 0:3]
        tpoints = np.matmul(trans, points.T).T
        return tpoints[:, :3]

    @staticmethod
    def deglobalize(pts_temp, trans):
        points = np.ones((pts_temp.shape[0], 4))
        points[:, 0:3] = pts_temp[:, 0:3]
        tpoints = np.matmul(trans, points.T).T
        return tpoints[:, :3]

    def __len__(self):
        return len(self.path_list)

    def get_dataset_stats(self):
        weights = np.zeros(self.learning_map.max()+1)

        for l in tqdm.tqdm(range(len(self.path_list)), desc='Loading weights', leave=True):
            pc_path = self.path_list[l]
            dir, file = os.path.split(pc_path)
            label_tmp = os.path.join(dir, '../labels', file[:-4] + '.npy')
            labels = np.load(label_tmp).astype(np.int32).reshape([-1])
            sem_labels = self.learning_map[labels]
            lbl, count = np.unique(sem_labels, return_counts=True)
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]

            weights[lbl] += count

        return weights


class PC2ImgConverter(object):

    def __init__(self, imgChannel=4, xRange=[0, 100], yRange=[-10, 10], zRange=[-10, 10], xGridSize=0.1, yGridSize=0.1,
                 zGridSize=0.1, num_classes=7):

        self.xRange = xRange
        self.yRange = yRange
        self.zRange = zRange
        self.xGridSize = xGridSize
        self.yGridSize = yGridSize
        self.zGridSize = zGridSize
        self.maxImgWidth = int((xRange[1] - xRange[0]) / xGridSize)
        self.maxImgHeight = int((yRange[1] - yRange[0]) / yGridSize)
        self.topViewImgDepth = int((zRange[1] - zRange[0]) / zGridSize)
        self.frontViewImgWidth = int((zRange[1] - zRange[0]) / zGridSize)
        self.frontViewImgHeight = int((yRange[1] - yRange[0]) / yGridSize)
        self.imgChannel = imgChannel
        self.maxDim = 5000
        self.num_classes = num_classes

    def getBEVImage(self, pointcloud, labels):

        """ top view x-y projection of the input point cloud"""
        """ max image size maxImgWidth=512 times maxImgHeight=64 """
        # topViewImage = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth, self.imgChannel), dtype=np.float32)
        #
        # imgMean = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth), dtype=np.float32)
        # imgMax = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth), dtype=np.float32)
        # imgDensity = np.zeros(shape=(self.maxImgHeight, self.maxImgWidth), dtype=np.float32)
        imgLabel = -np.ones(shape=(self.maxImgHeight, self.maxImgWidth), dtype=np.int32)
        imgPointsIdx = -np.ones(shape=(self.maxImgHeight, self.maxImgWidth), dtype=np.int32)
        # tempMatrix = np.empty(shape=(self.maxImgHeight, self.maxImgWidth, self.maxDim), dtype=np.float32)
        # tempMatrix[:] = np.nan
        # topViewPoints = []

        valid_labels = np.logical_not(labels == -1)

        valid_points = np.arange(pointcloud.shape[0])[valid_labels]

        valid_x = pointcloud[valid_points][:, 0]
        valid_y = pointcloud[valid_points][:, 1]
        valid_z = pointcloud[valid_points][:, 2]
        valid_l = labels[valid_labels]

        in_bound_x = np.logical_and(self.xRange[0] < valid_x, valid_x < self.xRange[1])
        in_bound_y = np.logical_and(self.yRange[0] < valid_y, valid_y < self.yRange[1])
        in_bound_z = np.logical_and(self.zRange[0] < valid_z, valid_z < self.zRange[1])
        in_bound_idx = np.logical_and(in_bound_x, np.logical_and(in_bound_y, in_bound_z))

        valid_points = valid_points[in_bound_idx]

        valid_x = -valid_x[in_bound_idx]
        valid_y = valid_y[in_bound_idx]
        valid_l = valid_l[in_bound_idx]

        pixel_x = np.floor((valid_x - self.xRange[0]) / self.xGridSize).astype(int)
        pixel_y = np.floor((valid_y - self.yRange[0]) / self.yGridSize).astype(int)

        imgLabel[pixel_y, pixel_x] = valid_l
        imgPointsIdx[pixel_y, pixel_x] = valid_points

        # imgLabel2 = -np.ones(shape=(self.maxImgHeight, self.maxImgWidth), dtype=np.int32)
        # # compute top view points
        # for p in range(0, len(pointcloud)):
        #
        #     xVal = pointcloud[p][0]
        #     yVal = pointcloud[p][1]
        #     zVal = pointcloud[p][2]
        #
        #     if self.xRange[0] < xVal < self.xRange[1] and self.yRange[0] < yVal < self.yRange[1] and self.zRange[0] < zVal < self.zRange[1]:
        #         topViewPoints.append(pointcloud[p])
        #         pixelX = np.int(np.floor((xVal - self.xRange[0]) / self.xGridSize))
        #         # pixelY = np.int(self.topViewImgHeight - np.floor((yVal - self.yRange[0]) / self.yGridSize))
        #         pixelY = np.int(np.floor((yVal - self.yRange[0]) / self.yGridSize))
        #         # imgDensity[pixelY, pixelX] += 1
        #         # indexVal = np.int(imgDensity[pixelY, pixelX])
        #         if labels[p] != -1:
        #             imgLabel2[pixelY, pixelX] = labels[p]
        #
        #         # if indexVal>= self.maxDim:
        #         #     print("ERROR in top view image computation: indexVal " + str(indexVal) + " is greater than maxDim " + str(self.maxDim))
        #         # tempMatrix[pixelY, pixelX, indexVal] = zVal

        # # compute statistics
        # for i in range(0, self.maxImgHeight):
        #     for j in range(0, self.maxImgWidth):
        #         currPixel = tempMatrix[i,j,:]
        #         currPixel = currPixel[~np.isnan(currPixel)]   # remove nans
        #
        #         if len(currPixel):
        #             imgMean[i,j] = np.mean(currPixel)
        #             imgMax[i,j] = np.max(currPixel)
        #
        # # convert to gray scale
        # grayMean = convertMean(imgMean)
        # grayMax = convertMean(imgMax)
        # grayDensity = convertDensity(imgDensity)

        # # place all computed images in a specific order
        # topViewImage[:, :, 0] = grayMean.astype(np.float32)
        # # topViewImage[:, :, 1] = grayRef.astype(np.float32)
        # # # topViewImage[:, :, 1] = grayMax.astype(np.float32)
        # # topViewImage[:, :, 2] = grayRef.astype(np.float32)
        # # topViewImage[:, :, 3] = grayDensity.astype(np.float32)
        # topViewCloud = np.asarray(topViewPoints)

        return imgLabel, imgPointsIdx, in_bound_idx

    def getBEVImageNew(self, pointcloud, labels):

        """ top view x-y projection of the input point cloud"""
        """ max image size maxImgWidth=512 times maxImgHeight=64 """

        imgLabel = -np.ones(shape=(self.maxImgHeight, self.maxImgWidth), dtype=np.int32)
        imgPointsIdx = -np.ones(shape=(self.maxImgHeight, self.maxImgWidth), dtype=np.int32)

        valid_labels = np.logical_not(labels == -1)

        valid_points = np.arange(pointcloud.shape[0])[valid_labels]

        valid_x = pointcloud[valid_points][:, 0]
        valid_y = pointcloud[valid_points][:, 1]
        valid_z = pointcloud[valid_points][:, 2]
        valid_l = labels[valid_labels]

        in_bound_x = np.logical_and(self.xRange[0] < valid_x, valid_x < self.xRange[1])
        in_bound_y = np.logical_and(self.yRange[0] < valid_y, valid_y < self.yRange[1])
        in_bound_z = np.logical_and(self.zRange[0] < valid_z, valid_z < self.zRange[1])
        in_bound_idx = np.logical_and(in_bound_x, np.logical_and(in_bound_y, in_bound_z))

        valid_points = valid_points[in_bound_idx]

        pixel_x = np.floor((valid_x[in_bound_idx] - self.xRange[0]) / self.xGridSize).astype(int)
        pixel_y = np.floor(self.maxImgHeight - (valid_y[in_bound_idx] - self.yRange[0]) / self.yGridSize).astype(int)-1
        # pixel_y = np.floor((valid_y[in_bound_idx] - self.yRange[0]) / self.yGridSize).astype(np.int)

        imgLabel[pixel_y, pixel_x] = valid_l[in_bound_idx]
        imgPointsIdx[pixel_y, pixel_x] = valid_points

        return imgLabel, imgPointsIdx

    def getCloudsFromBEVImage(self, predImg, topViewCloud, postProcessing = False):
        """ crop topviewcloud based on the network prediction image  """
        roadPoints = []
        vehPoints = []

        for p in range(0, len(topViewCloud)):

            xVal = topViewCloud[p][0]
            yVal = topViewCloud[p][1]
            zVal = topViewCloud[p][2]
            pixelX = int(np.floor((xVal - self.xRange[0]) / self.xGridSize))
            pixelY = int(self.topViewImgHeight - np.floor((yVal - self.yRange[0]) / self.yGridSize))
            classVal = predImg[pixelY, pixelX]


            if classVal == 1:
                roadPoints.append(topViewCloud[p])
            elif classVal == 2:
                vehPoints.append(topViewCloud[p])

        roadCloud = np.asarray(roadPoints)
        vehCloud = np.asarray(vehPoints)

        if postProcessing:

            # first global thresholding, make all points above 3  m as background
            globalThreshold = 3
            if len(roadCloud):
                roadCloud = roadCloud[roadCloud[:, 2] < globalThreshold]
            if len(vehCloud):
                vehCloud = vehCloud[vehCloud[:, 2] < globalThreshold]

            # second, apply thresholding only to road points
            # e.g. compute the mean of road points and remove those that are above
            if len(roadCloud):
                meanRoadZ = roadCloud[:, 2].mean()  # mean of third column, i.e. z values
                stdRoadZ = roadCloud[:, 2].std()  # mean of third column, i.e. z values
                roadThreshold = meanRoadZ + (1.0 * stdRoadZ)

                #print ("meanRoadZ: " + str(meanRoadZ) + " stdRoadZ: " + str(stdRoadZ) + " roadThreshold: " + str(roadThreshold))
                roadCloud = roadCloud[roadCloud[:, 2] < roadThreshold]

        return roadCloud, vehCloud


def sd_calc(data):
    n = len(data)

    if n <= 1:
        return 0.0

    mean, sd = avg_calc(data), 0.0

    # calculate stan. dev.
    for el in data:
        sd += (float(el) - mean)**2
    sd = math.sqrt(sd / float(n-1))

    return sd


def avg_calc(ls):
    n, mean = len(ls), 0.0

    if n <= 1:
        return ls[0]

    # calculate average
    for el in ls:
        mean = mean + float(el)
    mean = mean / float(n)

    return mean


def convertMean(input):
    output = input

    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            p = input[i, j]
            output[i,j] = MapHeightToGrayscale(p)

    return output


def MapHeightToGrayscale(currHeight):

    medianRoadHeight = -1.6
    minHeight = -3
    maxHeight = 3
    delta = (maxHeight - minHeight) / 256.0
    deltaHeight = currHeight - medianRoadHeight
    grayLevel = 0

    if deltaHeight >= maxHeight:
        grayLevel = 255
    elif deltaHeight <= minHeight:
        grayLevel = 0
    else:
        grayLevel = np.floor((deltaHeight - minHeight) / delta)

    if currHeight == 0:
        grayLevel = 0

    return grayLevel


def convertStd(input):
    output = input

    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            p = input[i, j]
            output[i,j] = MapStdToGrayscale(p)

    return output


def MapStdToGrayscale(std):

    minStd = 0
    maxStd = 1
    delta = (maxStd - minStd) / 256.0
    grayLevel = 0

    if std >= maxStd:
        grayLevel = 255
    elif std <= minStd:
        grayLevel = 0
    else:
        grayLevel = np.floor((std - minStd) / delta)

    return grayLevel


def convertDensity(input):
    output = input

    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            p = input[i, j]
            output[i,j] = MapDensityToGrayscale(p)

    return output


def MapDensityToGrayscale(density):

    minDensity = 0
    maxDensity = 16
    delta = (maxDensity - minDensity) / 256.0
    grayLevel = 0

    if density >= maxDensity:
        grayLevel = 255
    elif density <= minDensity:
        grayLevel = 0
    else:
        grayLevel = np.floor((density - minDensity) / delta)

    return grayLevel


def convertReflectivity(input):

    output = np.round(input*255)

    return output


class MultiBEVSourceDataset(Dataset):
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
        source_coordinates0 = source_data0['coordinates']
        source_coordinates1 = source_data1['coordinates']
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

        bev_labels0 = source_data0['bev_labels']
        bev_selected_idx0 = source_data0['bev_selected_idx']

        bev_labels1 = source_data1['bev_labels']
        bev_selected_idx1 = source_data1['bev_selected_idx']

        merged = dict(coordinates0=source_coordinates0,
                      coordinates1=source_coordinates1,
                      xyz0=source_xyz0,
                      xyz1=source_xyz1,
                      features0=source_features0,
                      features1=source_features1,
                      sem_labels0=source_sem_labels0,
                      sem_labels1=source_sem_labels1,
                      bev_labels0=bev_labels0,
                      bev_labels1=bev_labels1,
                      bev_selected_idx0=bev_selected_idx0,
                      bev_selected_idx1=bev_selected_idx1,
                      sampled_idx0=source_sampled_idx0,
                      sampled_idx1=source_sampled_idx1,
                      idx0=source_idx0,
                      idx1=source_idx1,
                      inverse_map0=source_inverse_map0,
                      inverse_map1=source_inverse_map1)
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

