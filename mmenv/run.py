import os
import cv2
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import defaultdict
from mmseg.registry import MODELS
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.dataset import Compose

from utils import test


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help="Select train or test mode")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    if not args.mode in ['train', 'test']:
        raise Exception("Select train or test mode")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")
    
    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    cfg.work_dir = "checkpoints"

    if args.mode == 'train':
        cfg.resume = False
        runner = Runner.from_cfg(cfg)
        runner.train()

    elif args.mode == 'test':
        model = MODELS.build(cfg.model)
        checkpoint = load_checkpoint(
            model,
            "/data/ephemeral/home/syp/level2-cv-semanticsegmentation-cv-20-lv3/mmenv/checkpoints/iter_20000.pth",
            map_location='cpu'
        )
        df_inference = test.main(cfg, model)
        df_inference.to_csv('submission.csv', index=False)