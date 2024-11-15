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
        "learning_rate": config['lr'],
        "max_epochs": config['num_epochs'],
        "batch_size": config['train_batch']
    }
    wandb.config.update(wandb_args)

    IMAGE_ROOT = config['train_img']
    LABEL_ROOT = config['train_label']
    SAVED_DIR = os.path.join(config['pt_saved_dir'], config['exp_name'])

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

    tf = TransformSelector(config['transform']['transform_type'], config['transform']["augmentations"]).get_transform()

    train_dataset = XRayDataset(pngs, jsons, IMAGE_ROOT, LABEL_ROOT, CLASSES, CLASS2IND, is_train=True, transforms=tf)
    valid_dataset = XRayDataset(pngs, jsons, IMAGE_ROOT, LABEL_ROOT, CLASSES, CLASS2IND,is_train=False, transforms=tf)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config['train_batch'],
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=config['val_batch'],
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    #model = models.segmentation.fcn_resnet50(pretrained=True)

    # output class 개수를 dataset에 맞도록 수정합니다.
    #model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

    model = modelSelector(config['model'], len(CLASSES)).get_model()

    # Loss function을 정의합니다.
    criterion = getattr(nn, config['criterion'])()

    # Optimizer를 정의합니다.
    optimizer = getattr(optim, config['optimizer'])(params=model.parameters(), lr=config['lr'], weight_decay=1e-6)

    set_seed(config['random_seed'])


    train(model, config['num_epochs'], CLASSES, train_loader, valid_loader, criterion, optimizer, config['val_every'], SAVED_DIR, config['model']['name'])

if __name__ == '__main__':
    main()