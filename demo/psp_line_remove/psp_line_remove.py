from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp

classes = ('R', 'G', 'B')
palette = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]

@DATASETS.register_module()
class LineRemoveDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.jpg', 
                     split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weighted_loss


@weighted_loss
def mse_loss(pred, target):
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


#### register mse loss function
@LOSSES.register_module()
class DltMSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0, loss_name='loss_dlt_mse'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # pred = pred - target # delta mse
        USE_MASK = True
        if USE_MASK:
            # reduction = 'sum'
            target_abs = target.abs()
            mask1 = target_abs > 1
            mask2 = mask1.logical_not()
            # mask2 = (target_abs > 0.2).logical_and(mask1.logical_not())
            # mask3 = target_abs <= 0.2
            loss1 = self.loss_weight * mse_loss(
                pred[mask1], target[mask1], weight, reduction=reduction, avg_factor=avg_factor)
            loss2 = self.loss_weight * mse_loss(
                pred[mask2], target[mask2], weight, reduction=reduction, avg_factor=avg_factor)
            # loss3 = self.loss_weight * mse_loss(
            #     pred[mask3], target[mask3], weight, reduction=reduction, avg_factor=avg_factor)
            # mask_num = (target.abs() > 1.0).sum()
            # eps = 1
            # loss = self.loss_weight * mse_loss(
            #     pred, target, weight, reduction=reduction, avg_factor=avg_factor)
            # return loss / (mask_num + eps)
            return loss1 + loss2
        else:
            loss = self.loss_weight * mse_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
            return loss

################
from mmseg.datasets.builder import PIPELINES
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.pipelines.formatting import to_tensor

@PIPELINES.register_module()
class LoadMyAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self, mean, std, to_rgb=True,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = mmcv.imnormalize(gt_semantic_seg.astype(np.float64)-results['img'].astype(np.float64), self.mean*0, self.std, self.to_rgb)
        # results['gt_semantic_seg'] = mmcv.imnormalize(gt_semantic_seg, self.mean, self.std, self.to_rgb)
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class MyFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg']).permute(2,0,1).contiguous(),
                stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__
################


from mmseg.apis import set_random_seed
from mmseg.utils import get_device
from mmcv import Config

file_root = '/home/andy/Downloads/private/mmsegmentation/demo/psp_line_remove'
## load config
cfg = Config.fromfile(osp.join(file_root, '../configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'))
# cfg = Config.fromfile(osp.join(file_root, '../configs/pspnet/pspnet_unet_s5_my.py'))

data_root = osp.join(file_root, '../data')
img_dir = 'img_dir'
ann_dir = 'ann_dir'
# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
# cfg.model.backbone.frozen_stages=5
cfg.model.auxiliary_head = None

# cfg.model.auxiliary_head.num_classes = 3
# cfg.model.auxiliary_head.loss_decode = dict(loss_weight=0.4, type='DltMSELoss')
# cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 3
cfg.model.pretrained = None
# cfg.model.auxiliary_head.num_classes = 8

# Modify dataset type and path
cfg.dataset_type = 'LineRemoveDataset'
cfg.data_root = data_root

cfg.data.samples_per_gpu =8
cfg.data.workers_per_gpu=1

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (384, 384)

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(type='LoadMyAnnotations', **cfg.img_norm_cfg),
    # dict(type='Resize', img_scale=(768, 768), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    # dict(type='DefaultFormatBundle'),
    dict(type='MyFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 240),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/datamy'

cfg.runner.max_iters = 1000
cfg.runner.type = 'IterBasedRunner'
cfg.log_config.interval = 10
cfg.evaluation.interval = 1e5
cfg.evaluation.pre_veal = False
cfg.checkpoint_config.interval = 50

cfg.optimizer.lr = 1e-4

cfg.model.decode_head.loss_decode = dict(loss_weight=1.0, type='DltMSELoss')

cfg.model.decode_head.num_classes = 3

cfg.model.backbone.pretrained = '/home/andy/Downloads/private/mmsegmentation/demo/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = get_device()
checkpoints_path = '/home/andy/Downloads/private/mmsegmentation/work_dirs/datamy/latest.pth'
cfg.load_from = checkpoints_path if osp.exists(checkpoints_path) else None

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
import mmcv

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
                meta=dict())