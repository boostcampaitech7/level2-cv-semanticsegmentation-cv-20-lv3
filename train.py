import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import XRayDataset
from function import set_seed, train
from custom_augments import TransformSelector
from selectModel import modelSelector
import wandb

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
    SAVED_DIR = os.path.join(config['paths']['model']['save_dir'], config['exp_name'])

    if not os.path.exists(SAVED_DIR):                                                           
        os.makedirs(SAVED_DIR)

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }

    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    pngs = sorted(pngs)
    jsons = sorted(jsons)

    tf = TransformSelector(config['transform']['type'], config['transform']["augmentations"]).get_transform()

    train_dataset = XRayDataset(pngs, jsons, IMAGE_ROOT, LABEL_ROOT, CLASSES, CLASS2IND, is_train=True, transforms=tf, debug=config['debug'])
    valid_dataset = XRayDataset(pngs, jsons, IMAGE_ROOT, LABEL_ROOT, CLASSES, CLASS2IND, is_train=False, transforms=tf, debug=config['debug'])
    

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
    optimizer = getattr(optim, config['optimizer']['type'])(params=model.parameters(), lr=config['optimizer']['lr'], weight_decay=1e-6)
    
    # Scheduler 설정
    scheduler = None
    if 'lr_scheduler' in config:
        scheduler_name = config['lr_scheduler']['type']  # 'CosineAnnealingLR' 또는 다른 스케줄러 종류
        scheduler_params = config['lr_scheduler']['params']  # 예: {'T_max': 10}
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_params)

    set_seed(config['random_seed'])

    train(model, config['training']['num_epochs'], CLASSES, train_loader, valid_loader, criterion, optimizer, config['training']['validate_every'], SAVED_DIR, config['model']['name'], config['model']['type'],
          config['early_stopping']['patience'], config['early_stopping']['delta'], scheduler = scheduler)

if __name__ == '__main__':
    main()
