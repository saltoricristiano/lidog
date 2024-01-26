import os
import torch
import yaml
import numpy as np
import tqdm
from torchvision.transforms import Compose
import math

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class NuScenesBEVDataset(BaseDataset):
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
                 use_bounds: bool = True,
                 decoder_2d_levels: list = ['block8'],
                 bev_img_sizes: dict = {'bottle': 240, 'block6': 240, 'block7': 240, 'block8': 240},
                 bound_2d: float = 50.0,
                 scaling_factor: dict = {'bottle': 1.0, 'block6': 1.0, 'block7': 1.0, 'block8': 1.0},
                 augment_bev: bool = True):

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

        self.name = 'NuScenesBEVDataset'

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
        self.use_bounds = use_bounds
        self.augment_bev = augment_bev
        self.decoder_2d_levels = decoder_2d_levels
        self.scaling_factor = scaling_factor
        self.bev_img_sizes = bev_img_sizes
        self.grid_bounds = [[-60, 60], [-60, 60], [-10, 8]]
        self.grid_bounds2d = [[-bound_2d, bound_2d], [-bound_2d, bound_2d], [-10, 8]]

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

    def __len__(self):
        return len(self.pcd_path)

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

    def __getitem__(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        if i not in self.CACHE.keys():

            pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 5))
            sem_labels = self.load_label_nusc(label_tmp)

            assert pcd.shape[0] == sem_labels.shape[0], f'Points and labels have shape {pcd.shape[0]} and {sem_labels.shape[0]}'

            points = pcd[:, :3]

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
            points, tr = self.augmentations(points)

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        if self.use_bounds:
            in_bound_idx = self.filter_bounds(points)

            points = points[in_bound_idx]
            sem_labels = sem_labels[in_bound_idx]
            colors = colors[in_bound_idx]

            sampled_idx = sampled_idx[in_bound_idx]

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

        bev_points = (quantized_coords*self.voxel_size).astype(np.float32)
        bev_labels = {}
        bev_selected_idx = {}

        for key in self.decoder_2d_levels:
            bev_labels_lvl, bev_selected_idx_lvl = self.bev_converters[key].getBEVImageNew(bev_points, sem_labels)

            bev_labels[key] = torch.from_numpy(bev_labels_lvl).long()
            bev_selected_idx[key] = torch.from_numpy(bev_selected_idx_lvl).long()

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
                "inverse_map": inverse_map,
                "bev_labels": bev_labels,
                "bev_selected_idx": bev_selected_idx}

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


class PC2ImgConverter(object):

    def __init__(self, imgChannel=4, xRange=[0, 100], yRange=[-10, 10], zRange=[-10, 10], xGridSize=0.1, yGridSize=0.1,
                 zGridSize=0.1, num_classes=7):

        self.xRange = xRange
        self.yRange = yRange
        self.zRange = zRange
        self.xGridSize = xGridSize
        self.yGridSize = yGridSize
        self.zGridSize = zGridSize
        self.maxImgWidth = np.int((xRange[1] - xRange[0]) / xGridSize)
        self.maxImgHeight = np.int((yRange[1] - yRange[0]) / yGridSize)
        self.topViewImgDepth = np.int((zRange[1] - zRange[0]) / zGridSize)
        self.frontViewImgWidth = np.int((zRange[1] - zRange[0]) / zGridSize)
        self.frontViewImgHeight = np.int((yRange[1] - yRange[0]) / yGridSize)
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

        pixel_x = np.floor((valid_x[in_bound_idx] - self.xRange[0]) / self.xGridSize).astype(np.int)
        pixel_y = np.floor((valid_y[in_bound_idx] - self.yRange[0]) / self.yGridSize).astype(np.int)

        imgLabel[pixel_y, pixel_x] = valid_l[in_bound_idx]
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

        pixel_x = np.floor((valid_x[in_bound_idx] - self.xRange[0]) / self.xGridSize).astype(np.int)
        pixel_y = np.floor(self.maxImgHeight - (valid_y[in_bound_idx] - self.yRange[0]) / self.yGridSize).astype(np.int)-1
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
            pixelX = np.int(np.floor((xVal - self.xRange[0]) / self.xGridSize))
            pixelY = np.int(self.topViewImgHeight - np.floor((yVal - self.yRange[0]) / self.yGridSize))
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
