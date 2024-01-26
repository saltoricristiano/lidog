# **Walking Your LiDOG🐶: A Journey Through Multiple Domains for LiDAR Semantic Segmentation [ICCV2023]**

The official implementation of our work "Walking Your LiDOG: A Journey Through Multiple Domains for LiDAR Semantic Segmentation".

## Introduction
The ability to deploy robots that can operate safely in diverse environments is crucial for developing embodied intelligent agents. 
As a community, we have made tremendous progress in within-domain LiDAR semantic segmentation. 
However, do these methods generalize across domains? 
To answer this question, we design the first experimental setup for studying domain generalization (DG) for LiDAR semantic segmentation (DG-LSS). 
Our results confirm a significant gap between methods, evaluated in a cross-domain setting: for example, a model trained on the source dataset (SemanticKITTI) obtains 26.53 mIoU on the target data, compared to 48.49 mIoU obtained by the model trained on the target domain (nuScenes).
To tackle this gap, we propose the first method specifically designed for DG-LSS, which obtains 34.88 mIoU on the target domain, outperforming all baselines. 
Our method augments a sparse-convolutional encoder-decoder 3D segmentation network with an additional, dense 2D convolutional decoder that learns to classify a birds-eye view of the point cloud. 
This simple auxiliary task encourages the 3D network to learn features that are robust to sensor placement shifts and resolution, and are transferable across domains. 
With this work, we aim to inspire the community to develop and evaluate future models in such cross-domain conditions.

:fire: For more information follow the [PAPER](https://arxiv.org/abs/2304.11705) link!:fire:

Authors: [Cristiano Saltori](https://saltoricristiano.github.io),
         [Aljoša Ošep](https://aljosaosep.github.io),
         [Elisa Ricci](https://scholar.google.ca/citations?user=xf1T870AAAAJ&hl),
         [Laura Leal-Taixé](https://scholar.google.com/citations?user=tT2TC-UAAAAJ&hl=en)

![teaser](assets/teaser.png)

## Installation (Pip/Venv/Conda)
The code has been tested with Python v3.8, CUDA v11.1, pytorch v1.8.2 and pytorch-lighting v1.6.4.
Any other version may require to update the code for compatibility.
In your virtual environment install:
- [MinkowskiEngine v0.5.4](https://github.com/NVIDIA/MinkowskiEngine). This will install all the required packages (torch v1.8.2) together with the MinkowskiEngine.
- [pytorch-lightning v1.6.4](https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix)
- [open3d v0.10.0](http://www.open3d.org)
- [wandb v0.12.18](https://docs.wandb.ai/quickstart)
- [nuscenes-devkit v1.1.9](https://github.com/nutonomy/nuscenes-devkit)


## Data preparation

### Synth4D
To download Synth4D follow the instructions [here](https://github.com/saltoricristiano/gipso-sfouda/tree/main). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──kitti_synth/
    |   ├──Town03/
    |   |     ├── calib/
    |   |     |    ├── 000000.npy
    |   |     |    └── ... 
    |   |     ├── labels/
    |   |     |    ├── 000000.npy
    |   |     |    └── ...
    |   |     └── velodyne/
    |   |          ├── 000000.npy
    |   |          └── ...
    |   ├──Town06/
    |   ├──Town07/
    |   └──Town10HD/
    ├──nuscenes_synth/
    └──splits/
```

### SemanticKITTI
To download SemanticKITTI follow the instructions [here](http://www.semantic-kitti.org). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── times.txt
            └── 08/
```

### nuScenes
To download nuScenes-lidarseg follow the instructions [here](https://www.nuscenes.org). Paths will be already in the correct order.

### Soft-links
After you downloaded the datasets you need, create soft-links in the ```datasets``` directory
```
cd lidog
mkdir datasets
cd datasets
ln -s PATH/TO/SEMANTICKITTI SemanticKITTI
# do the same for the other datasets
```
Alternatively, change the path in the config files.

## LiDOG🐶
Training LiDOG is as easy as running the command:
```
python train_lidog.py --config_file your/config/file/path
```
For example, for training LiDOG on Synth4D-kitti run:
```
python train_lidog.py --config_file configs/lidog/single/synth4d-kitti.yaml
```
All the configuration files use 4xGPUS. You may need to change the batch size and number of GPUs according to your computational capabilities.

## Baselines
We provide the codebase for running all our baselines. Similarly to LiDOG, for training our baselines run:
```
python train/baseline/script.py --config_file configs/baseline/dataset.yaml
```
For example, if you want to run mix3D on Synth4D-kitti run:
```
python train_aug_based.py --config-file configs/mix3D/single/synth4d.yaml
```
All the configuration files use 4xGPUS. You may need to change the batch size and number of GPUs according to your computational capabilities.

## Lookup table
According to the type of baseline, we use a different training script. We provide each setting in the following table.

| Method  | Script                 |
| ------------- |------------------------|
| Source/Target  | train_source.py        |
| Mix3D | train_aug_based.py     |
| CoSMix  | train_aug_based.py     |
| PointCutMix  | train_aug_based.py     |
| IBN  | train_source.py        |
| RobustNet  | train_source.py        |
| SN  | train_scaling_based.py |
| Raycast  | train_source.py        |
| LiDOG  | train_lidog.py         |



## Evaluation
To evaluate a model after training, run:
```
python eval_target.py --config_file configs/of/your/model --resume_checkpoint path/to/your/model.ckpt
```
You can save predictions for future visualizations by adding ```--save_predictions```.
This will save a 

## References
If you use our work, please cite us:
```
@inproceedings{saltori2023walking,
  title={Walking Your LiDOG: A Journey Through Multiple Domains for LiDAR Semantic Segmentation},
  author={Saltori, Cristiano and Osep, Aljosa and Ricci, Elisa and Leal-Taix{\'e}, Laura},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## Acknowledgments
This project was partially funded by the Sofja Kovalevskaja Award of the Humboldt Foundation,
the EU ISFP project PRE- CRISIS (ISFP-2022-TFI-AG-PROTECT-02-101100539),
the PRIN project LEGO-AI (Prot. 2020TA3K9N) and the MUR PNRR project FAIR - Future AI Research (PE00000013) funded by the NextGenerationEU.
It was carried out in the Vision and Learning joint laboratory of FBK-UNITN and used the CINECA, NVIDIA-AI TC clusters to run part of the experiments.
We thank T. Meinhardt and I. Elezi for their feedback on this manuscript. 
The authors of this work take full responsibility for its content.

## Thanks

We thank the opensource project [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).