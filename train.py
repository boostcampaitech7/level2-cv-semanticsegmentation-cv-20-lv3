import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tools.custom_dataset import XRayDataset
from tools.function import set_seed, train
from tools.custom_augments import TransformSelector
from tools.select_model import modelSelector
import wandb

def check_data(png_prefix, jsons_prefix):
    png_prefix = sorted(png_prefix)
    jsons_prefix = sorted(jsons_prefix)
    for png, json in zip(png_prefix, jsons_prefix):
        if png != json:
            print(png,json)


def load_data(IMAGE_ROOT, LABEL_ROOT, DATA_ROOT = './data', num = None):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=DATA_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=DATA_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }

    
    jsons_fn_prefix = {os.path.splitext(os.path.split(fname)[-1])[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(os.path.split(fname)[-1])[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
    if num:
        pngs = sorted(pngs)[:num]
        jsons = sorted(jsons)[:num]
        pngs = set(pngs)
        jsons = set(jsons)
    return pngs, jsons

def main(config, CLASSES, CLASS2IND):
    
    wandb.init(project=config['proj_name'])
    wandb.run.name = config['exp_name']
    wandb.run.save()
    wandb_args = {
        "learning_rate": config['optimizer']['lr'],
        "max_epochs": config['training']['num_epochs'],
        "batch_size": config['training']['batch_size']['train']
    }
    wandb.config.update(wandb_args)

    IMAGE_ROOT = config['paths']['train']['image']
    LABEL_ROOT = config['paths']['train']['label']
    DATA_ROOT = config['paths']['data']
    SAVED_DIR = os.path.join(config['paths']['model']['save_dir'], config['exp_name'])

    if not os.path.exists(SAVED_DIR):                                                           
        os.makedirs(SAVED_DIR)

    pngs, jsons = load_data(IMAGE_ROOT, LABEL_ROOT, DATA_ROOT)

    if config['pseudo_labeling']['enabled']:
        TEST_IMAGE_ROOT = config['paths']['test']['image']
        TEST_LABEL_ROOT = config['pseudo_labeling']['pseudo_dir']
        test_pngs, test_jsons = load_data(TEST_IMAGE_ROOT, TEST_LABEL_ROOT, DATA_ROOT, num = config['pseudo_labeling']['max_pseudo_samples'])
        pngs.update(test_pngs)
        jsons.update(test_jsons)

    pngs = sorted(pngs)
    jsons = sorted(jsons)

    cropped_pngs = pngs[800:]
    cropped_jsons = jsons[800:]

    pngs = pngs[:800]
    jsons = jsons[:800]
    tft = None
    tfv = None

    if config['transform']['train'] is not None:
        tft = TransformSelector(config['transform']['train']['type'], config['transform']['train']["augmentations"]).get_transform()
    if config['transform']['val'] is not None:
        tfv = TransformSelector(config['transform']['val']['type'], config['transform']['val']["augmentations"]).get_transform()
    
    
    train_dataset = XRayDataset(pngs, jsons, cropped_pngs, cropped_jsons, DATA_ROOT, CLASSES, CLASS2IND, is_train=True, transforms=tft, debug=config['debug'], cropped=config['cropped'])
    valid_dataset = XRayDataset(pngs, jsons, cropped_pngs, cropped_jsons, DATA_ROOT, CLASSES, CLASS2IND, is_train=False, transforms=tfv, debug=config['debug'])
    print(len(train_dataset), len(valid_dataset))
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config['training']['batch_size']['train'],
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=config['training']['batch_size']['val'],
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    model = modelSelector(config['model'], len(CLASSES)).get_model()

    # Loss function을 정의합니다.
    criterion = getattr(nn, config['criterion'])()

    # Optimizer를 정의합니다.
    if config['optimizer']['type'] == 'Lion':
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=config['optimizer']['lr'], weight_decay=1e-6)
    else:
        optimizer = getattr(optim, config['optimizer']['type'])(params=model.parameters(), lr=config['optimizer']['lr'], weight_decay=1e-6)
    
    # Scheduler 설정
    scheduler = None
    if 'lr_scheduler' in config:
        scheduler_name = config['lr_scheduler']['type']  # 'CosineAnnealingLR' 또는 다른 스케줄러 종류
        scheduler_params = config['lr_scheduler']['params']  # 예: {'T_max': 10}
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_params)

    set_seed(config['random_seed'])

    train(model, config['training']['num_epochs'], CLASSES, train_loader, valid_loader, criterion, optimizer, config['training']['validate_every'], SAVED_DIR, config['model']['name'], config['model']['type'], config['model']['arch'],
          config['early_stopping']['patience'], config['early_stopping']['delta'], scheduler = scheduler)
