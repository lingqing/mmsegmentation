from mmseg.apis import set_random_seed
from mmseg.utils import get_device
from mmcv import Config
import os.path as osp

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint

classes = ('R', 'G', 'B')
palette = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]

file_root = '/home/andy/Downloads/private/mmsegmentation/demo/psp_line_remove'
## load config
cfg = Config.fromfile(osp.join(file_root, '../configs/pspnet/fcn_unet_s5_my.py'))
cfg.model.decode_head.loss_decode = dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False)
# cfg.norm_cfg = dict(type='BN', requires_grad=True)
# cfg.model.backbone.norm_cfg = cfg.norm_cfg
# cfg.model.decode_head.norm_cfg = cfg.norm_cfg
# cfg.model.auxiliary_head = None
# cfg.model.decode_head.num_classes = 3
# # cfg.model.pretrained = None
# cfg.img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

checkpoint_file='/home/andy/Downloads/private/mmsegmentation/work_dirs/datamy/latest.pth'
model = build_segmentor(cfg.model)
load_checkpoint(model, checkpoint_file, map_location='cpu')

model.CLASSES = classes
model.PALETT = palette
model.cfg = cfg  # save the config in the model for convenience
model.to('cuda:0')
model.eval()

img= file_root + '/../data/img_dir/1655988343.038572073.jpg'

from mmcv.parallel import collate, scatter
# from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
import mmcv, torch

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

cfg.crop_size = (384, 384)
cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadMyAnnotations', **cfg.img_norm_cfg),
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=(320, 240),
    #     img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    #     flip=False,
    #     transforms=[
            # dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            # dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
            dict(type='Normalize', **cfg.img_norm_cfg),
            # dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'img_norm_cfg')),
        # ]
        # )
]

# import torch
imgs = [img]
device = next(model.parameters()).device  # model device
# build the data pipeline
test_pipeline = [LoadImage()] + cfg.test_pipeline[1:]
# test_pipeline = cfg.test_pipeline
test_pipeline = Compose(test_pipeline)

data = []
for img in imgs:
    img_data = dict(img=img)
    img_data = test_pipeline(img_data)
    data.append(img_data)
data = collate(data, samples_per_gpu=len(imgs))
data = scatter(data, [device])[0]
with torch.no_grad():
    result = model.whole_inference(data['img'], img_meta=data['img_metas'], rescale=False)
    # result = model.slide_inference(data['img'][0], img_meta=data['img_metas'][0], rescale=True)

# import matplotlib.pylab as plt
# plt.figure(figsize=(18,16))
# plt.subplot(1,2,1)
ori_img = mmcv.imread(img)
# plt.imshow(ori_img)
# plt.show(result)
import numpy as np
# plt.subplot(1,2,2)

result_img = np.array(result.squeeze(0).cpu()).transpose(1,2,0) * cfg.img_norm_cfg['std']
result_img = result_img[:,:,::-1]
# result_img = np.array(result.squeeze(0).cpu()).transpose(1,2,0)
# plt.imshow((result_img).astype(np.uint8))
print(f'max={result_img.max()}, min={result_img.min()}')
# plt.show()
import cv2
cv2.imwrite('result.jpg', (-result_img+cfg.img_norm_cfg['mean']).astype(np.uint8))
cv2.imwrite('result+ori.jpg', (result_img+ori_img).astype(np.uint8))
