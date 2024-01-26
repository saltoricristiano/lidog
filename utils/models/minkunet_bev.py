import torch
import MinkowskiEngine as ME
import torch.nn as nn
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from utils.models.conv2d import Encoder2D
import numpy as np


class MinkUNetBaseBEV(nn.Module):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self,
                 in_channels,
                 out_channels,
                 D,
                 initial_kernel_size=5,
                 dynamic_mapping=False,
                 decoder_2d_level=['block8'],
                 bottle_img_dim={'block8': 128, 'block7': 64, 'block6': 64, 'bottle': 64},
                 bottle_out_img_dim={'block8': 480, 'block7': 240, 'block6': 240, 'bottle': 240},
                 mapping_bound_2d=50.0,
                 scaling_factors={'block8': 1.0, 'block7': 1.0, 'block6': 1.0, 'bottle': 1.0},
                 binary_seg_layer=False):

        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None
        self.mapping_bound_2d = mapping_bound_2d
        self.binary_seg = binary_seg_layer
        self.network_initialization(in_channels, out_channels, D, initial_kernel_size,
                                    dynamic_mapping, scaling_factors, decoder_2d_level, bottle_img_dim,
                                    bottle_out_img_dim, binary_seg_layer)
        self.weight_initialization()

    def network_initialization(self, in_channels,
                               out_channels,
                               D,
                               initial_kernel_size=5,
                               dynamic_mapping=False,
                               scaling_factors={'block8': 1.0, 'block7': 1.0, 'block6': 1.0, 'bottle': 1.0},
                               decoder_2d_level=['block8'],
                               bottle_img_dim={'block8': 128, 'block7': 64, 'block6': 64, 'bottle': 64},
                               bottle_out_img_dim={'block8': 480, 'block7': 240, 'block6': 240, 'bottle': 240},
                               binary_seg=False):

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=initial_kernel_size, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.dropout = ME.MinkowskiDropout(p=0.5)

        self.pool2D = nn.MaxPool2d(5, 3, 1)

        self.relu_2d = nn.ReLU()

        self.dynamic_mapping = dynamic_mapping

        self.mapping_boundaries = [[-self.mapping_bound_2d, self.mapping_bound_2d],
                                   [-self.mapping_bound_2d, self.mapping_bound_2d],
                                   [-10, 8]]
        self.decoder_block_out_sizes = {'block8': 96, 'block7': 96, 'block6': 128, 'bottle': 256}
        self.decoder_2d_level = decoder_2d_level
        self.bottle_img_dim = bottle_img_dim
        self.bottle_out_img_dim = bottle_out_img_dim
        self.scaling_factors = scaling_factors

        encoders2d = {}
        for k in self.decoder_2d_level:

            encoders2d[k] = Encoder2D(input_size=self.decoder_block_out_sizes[k],
                                      n_classes=out_channels,
                                      binary_seg=binary_seg)
        self.encoders2d = nn.ModuleDict(encoders2d)

        scaled_pool2d = {}
        for k in self.decoder_2d_level:
            scaled_stride = int(3/self.scaling_factors[k])
            scaled_pool2d[str(int(self.scaling_factors[k]*100))] = nn.MaxPool2d(5, scaled_stride, 1)

        self.scaled_pool2d = nn.ModuleDict(scaled_pool2d)

    def filter_bounds(self, points):

        pts_x = points[:, 0]
        pts_y = points[:, 1]

        in_bound_x = torch.logical_and(self.mapping_boundaries[0][0] < pts_x, pts_x < self.mapping_boundaries[0][1])
        in_bound_y = torch.logical_and(self.mapping_boundaries[1][0] < pts_y, pts_y < self.mapping_boundaries[1][1])
        in_bound_idx = torch.logical_and(in_bound_x, in_bound_y)

        return in_bound_idx

    def sparse2super(self, x, input_voxel_size=0.05, scaling_factor=1.0):
        batched_bev_feat_maps = []

        batch_bottle_coords, batch_bottle_feats = x.C, x.F
        # batch_bottle_feats = self.relu_2d(batch_bottle_feats)
        batch_bottle_coords = batch_bottle_coords.cpu()
        batch_bottle_idx = batch_bottle_coords[:, 0]
        batch_bottle_xyz = batch_bottle_coords[:, 1:] * input_voxel_size

        quantized_min_x = self.mapping_boundaries[0][0]
        quantized_max_x = self.mapping_boundaries[0][1]
        quantized_min_y = self.mapping_boundaries[1][0]
        quantized_max_y = self.mapping_boundaries[1][1]


        max_height = torch.tensor((quantized_max_y - quantized_min_y) / input_voxel_size).int()
        max_width = torch.tensor((quantized_max_x - quantized_min_x) / input_voxel_size).int()


        feat_size = batch_bottle_feats.shape[-1]

        """ top view x-y projection of the input point cloud"""
        """ max image size maxImgWidth=512 times maxImgHeight=64 """

        for b in range(batch_bottle_idx.max()+1):

            b_bottle_idx = batch_bottle_idx == b
            # features of the bottleneck points
            bottle_b_feats = batch_bottle_feats[b_bottle_idx]
            # coordinates of the bottleneck points
            bottle_b_coords = batch_bottle_xyz[b_bottle_idx]

            # due to approx some points may be on the border
            in_bounds_idx = self.filter_bounds(bottle_b_coords)

            bottle_b_feats = bottle_b_feats[in_bounds_idx]
            bottle_b_coords = bottle_b_coords[in_bounds_idx]

            valid_x = bottle_b_coords[:, 0]
            valid_y = bottle_b_coords[:, 1]
            valid_z = bottle_b_coords[:, 2]

            b_bev_feats = torch.zeros((max_height, max_width, feat_size)).to(x.device)

            pixel_x = torch.floor((valid_x - quantized_min_x) / input_voxel_size).long()
            pixel_y = torch.floor(max_height - (valid_y - quantized_min_y) / input_voxel_size).long() - 1
            # pixel_z = torch.floor((valid_z - quantized_min_z) / input_voxel_size).long().to(x.device)

            b_bev_feats[pixel_y.to(x.device), pixel_x.to(x.device)] = bottle_b_feats

            if scaling_factor == 1.0:
                # with voxel_size 0.05 and boundaries [-50, 50], b_bev_feats hape is [666, 666]
                b_bev_feats = self.pool2D(b_bev_feats.view(1, -1, max_height, max_width))
            else:
                # if lower res is used we scale the max pooling
                b_bev_feats = self.scaled_pool2d[str(int(scaling_factor*100))](b_bev_feats.view(1, -1, max_height, max_width))

            batched_bev_feat_maps.append(b_bev_feats.to(x.device))

        batched_bev_feat_maps = torch.cat(batched_bev_feat_maps, dim=0)

        return batched_bev_feat_maps

    def sparse2dense(self, x, input_xyz, bottle_img_dim=64, input_voxel_size=0.05):
        batched_bev_feat_maps = []

        batch_bottle_coords, batch_bottle_feats = x.C, x.F
        batch_bottle_coords = batch_bottle_coords.cpu()
        batch_bottle_idx = batch_bottle_coords[:, 0]
        batch_bottle_xyz = batch_bottle_coords[:, 1:]

        batch_input = input_xyz.cpu()
        batch_input = batch_input.cpu()
        batch_input_idx = batch_input[:, 0]
        batch_input_xyz = batch_input[:, 1:]

        feat_size = batch_bottle_feats.shape[-1]

        for b in range(batch_bottle_idx.max()+1):
            b_bottle_idx = batch_bottle_idx == b
            # features of the bottleneck points
            bottle_b_feats = batch_bottle_feats[b_bottle_idx]
            # coordinates of the bottleneck points
            bottle_b_coords = batch_bottle_xyz[b_bottle_idx] * input_voxel_size

            b_input_idx = batch_input_idx == b
            b_input_xyz = batch_input_xyz[b_input_idx]

            if self.dynamic_mapping:
                quantized_min_x = b_input_xyz[:, 0].min() * input_voxel_size
                quantized_max_x = b_input_xyz[:, 0].max() * input_voxel_size
                quantized_min_y = b_input_xyz[:, 1].min() * input_voxel_size
                quantized_max_y = b_input_xyz[:, 1].max() * input_voxel_size
                quantized_min_z = b_input_xyz[:, 2].min() * input_voxel_size
                quantized_max_z = b_input_xyz[:, 2].max() * input_voxel_size
            else:
                quantized_min_x = self.mapping_boundaries[0][0]
                quantized_max_x = self.mapping_boundaries[0][1]
                quantized_min_y = self.mapping_boundaries[1][0]
                quantized_max_y = self.mapping_boundaries[1][1]
                quantized_min_z = self.mapping_boundaries[2][0]
                quantized_max_z = self.mapping_boundaries[2][1]

            x_ptp = abs(quantized_min_x) + quantized_max_x
            y_ptp = abs(quantized_min_y) + quantized_max_y
            z_ptp = abs(quantized_min_z) + quantized_max_z
            quantization_x = np.ceil(x_ptp / bottle_img_dim)+1
            quantization_y = np.ceil(y_ptp / bottle_img_dim)+1
            quantization_z = np.ceil(z_ptp)+1

            quantized_bottle_coords, _, inverse_bottle_map = ME.utils.sparse_quantize(bottle_b_coords.numpy(),
                                                                                            quantization_size=[quantization_x,
                                                                                                               quantization_y,
                                                                                                               quantization_z],
                                                                                            return_index=True,
                                                                                            return_inverse=True)

            quantized_bottle_coords = quantized_bottle_coords[:, :2]

            quantized_bottle_coords[:, 0] = quantized_bottle_coords[:, 0] + int(bottle_img_dim/2)
            quantized_bottle_coords[:, 1] = quantized_bottle_coords[:, 1] + int(bottle_img_dim/2)
            bev_map = torch.zeros([bottle_img_dim, bottle_img_dim, feat_size]).to(x.device)
            for v_idx in range(inverse_bottle_map.max()):
                voxel_idx = inverse_bottle_map == v_idx
                voxel_feats = bottle_b_feats[voxel_idx].mean(dim=0)
                voxel_center = quantized_bottle_coords[v_idx]

                bev_map[int(voxel_center[0]), int(voxel_center[1])] = voxel_feats
            batched_bev_feat_maps.append(bev_map.view(1, -1, bottle_img_dim, bottle_img_dim))
        batched_bev_feat_maps = torch.cat(batched_bev_feat_maps, dim=0)

        return batched_bev_feat_maps

    def forward(self, x, is_seg=True, is_train=False):
        input_x = x

        input_bev_feats = {}

        out = self.conv0p1s1(input_x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_bottle = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out_bottle)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out_block5 = self.block5(out)

        if 'bottle' in self.decoder_2d_level:
            input_bev_feats['bottle'] = out_block5

        # tensor_stride=4
        out = self.convtr5p8s2(out_block5)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out_block6 = self.block6(out)

        if 'block6' in self.decoder_2d_level:
            input_bev_feats['block6'] = out_block6

        # tensor_stride=2
        out = self.convtr6p4s2(out_block6)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out_block7 = self.block7(out)

        if 'block7' in self.decoder_2d_level:
            input_bev_feats['block7'] = out_block7

        # tensor_stride=1
        out = self.convtr7p2s2(out_block7)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out_block8 = self.block8(out)

        if 'block8' in self.decoder_2d_level:
            input_bev_feats['block8'] = out_block8

        if is_train:

            img_pred = {}

            for key in self.encoders2d.keys():

                bev_feats_lvl = self.sparse2super(x=input_bev_feats[key], scaling_factor=self.scaling_factors[key])

                if not self.binary_seg:
                    img_pred[key] = self.encoders2d[key](bev_feats_lvl)
                else:
                    img_pred_tmp = self.encoders2d[key](bev_feats_lvl)
                    img_pred[key] = img_pred_tmp[0]
                    img_pred[key+'_binary'] = img_pred_tmp[1]

        else:
            bev_feat_map = None
            img_pred = None

        if is_seg:
            # return self.final(out_block8), out_block8
            return self.final(out_block8), img_pred
        else:
            return self.final(out_block8), img_pred, out_bottle, bev_feat_map

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)


class MinkUNet34BEV(MinkUNetBaseBEV):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

