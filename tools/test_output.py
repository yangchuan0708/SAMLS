import cv2
import torch
import os
import glob
import mmcv
import mmengine
import numpy as np
from mmengine import Config, get
from mmengine.dataset import Compose
from mmpl.registry import MODELS, VISUALIZERS
from mmpl.utils import register_all_modules
register_all_modules()
import mmengine.fileio as fileio

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def construct_sample(img, pipeline):
   # img = np.array(img)[:, :, ::-1]
   img = np.array(img)
   inputs = {
      'ori_shape': img.shape[:2],
      'img': img,
   }
   pipeline = Compose(pipeline)
   sample = pipeline(inputs)
   return sample


def build_model(cp, model_cfg):
   model_cpkt = torch.load(cp, map_location='cpu')
   model = MODELS.build(model_cfg)
   model.load_state_dict(model_cpkt["state_dict"], strict=True)
   model.to(device=device)
   model.eval()
   return model


def inference_func(ori_img, model, cfg, img_name):

   predict_pipeline = [
      # dict(type='mmdet.Resize', scale=(1024, 1024)),
      dict(type='mmdet.Resize', scale=(256, 256)),
      dict(
         type='mmdet.PackDetInputs',
         meta_keys=('ori_shape', 'img_shape', 'scale_factor'))
   ]

   # predict_pipeline = [
   #    dict(type='mmdet.LoadImageFromFile', backend_args=None),
   #
   #    dict(type='mmdet.Resize', scale=(256, 256)),
   #    # If you don't have a gt annotation, delete the pipeline
   #    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
   #    dict(
   #      type='mmdet.PackDetInputs',
   #      meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
   #                 'scale_factor'))]



   sample = construct_sample(ori_img, predict_pipeline)
   sample['inputs'] = [sample['inputs']]
   sample['data_samples'] = [sample['data_samples']]

   with torch.no_grad():
      pred_results = model.predict_step(sample, batch_idx=0)

   cfg.visualizer.setdefault('save_dir', 'visualizer')
   visualizer = VISUALIZERS.build(cfg.visualizer)

   data_sample = pred_results[0]
   img = np.array(ori_img).copy()
   out_file = f'visualizer/{img_name}'
   print(out_file)
   mmengine.mkdir_or_exist(os.path.dirname(out_file))

   visualizer.add_datasample(
      'test_img',
      img,
      draw_gt=False,
      data_sample=data_sample,
      show=False,
      wait_time=0.01,
      pred_score_thr=0.8,
      out_file=out_file
   )
   img_bytes = get(out_file)
   img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
   return img


# checkpoint = r'D:\RSPrompter-cky\tools\results\bijie_ins\E20230629_1\checkpoints\noadaption_epoch_epoch=9-map_valsegm_map_0=0.6630.ckpt'
# cfg = r'D:\RSPrompter-cky\configs\rsprompter\rsprompter_anchor_bijie_config.py'
# checkpoint = r"D:\RSPrompter-cky\tools\results\bijie_mask2former_ins\E20230525_1\checkpoints\epoch_epoch=5-map_valmap_0=0.0000.ckpt"
# cfg = r"D:\RSPrompter-cky\configs\rsprompter\mask2former_whu_config.py"
# checkpoint = r"D:\RSPrompter-cky\tools\results\bijie_hillshade_ins\E20230629_1\checkpoints\epoch_epoch=41-map_valsegm_map_0=0.0690.ckpt"
# cfg = r"D:\RSPrompter-cky\configs\rsprompter\rsprompter_anchor_bijie_dem_config.py"
# checkpoint = r"D:\RSPrompter-cky\tools\results\bijie_rgbd_ins\E20230629_1\checkpoints\epoch_epoch=47-map_valsegm_map_0=0.4950.ckpt"
# cfg = r"D:\RSPrompter-cky\configs\rsprompter\rsprompter_anchor_bijie_config.py"
# checkpoint = r"D:\RSPrompter-cky\tools\results\bijie_maskrcnn_ins\E20230525_0\checkpoints\epoch_epoch=5-map_valmap_0=0.0000.ckpt"
# cfg = r"D:\RSPrompter-cky\configs\rsprompter\maskrcnn_whu_config.py"
checkpoint = r"D:\RSPrompter-cky\tools\results\bijie_rgbd_samus_ins\E20230629_2\checkpoints\epoch_epoch=14-map_valsegm_map_0=0.5430.ckpt"
cfg = r"D:\RSPrompter-cky\configs\rsprompter\rsprompter_anchor_bijie_config.py"
# checkpoint = r"D:\RSPrompter-cky\tools\results\whu_ins\E20230629_1\checkpoints\epoch_epoch=14-map_valsegm_map_0=0.6290.ckpt"
# cfg = r"D:\RSPrompter-cky\configs\rsprompter\rsprompter_anchor_whu_config.py"
cfg = Config.fromfile(cfg)

model = build_model(checkpoint, cfg.model_cfg)

img_dir = r"D:\RSPrompter-cky\data\bijie_rgbd\val2017\image"
# img_dir = r"D:\RSPrompter-cky\data\bijie\val2017\image"
for img_name in os.listdir(img_dir):
   # img = cv2.imread(os.path.join(img_dir, img_name))
   img_bytes = fileio.get(os.path.join(img_dir, img_name), backend_args=None)
   img = mmcv.imfrombytes(img_bytes, flag='color', backend='tifffile')
   dst_img = inference_func(img, model, cfg, img_name)