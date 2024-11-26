import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from custom_dataset import XRayDataset
from custom_augments import TransformSelector
from selectModel import modelSelector
import optuna
from optuna.samplers import TPESampler
import wandb
from transformers import AutoImageProcessor

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

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def validation(model, data_loader, criterion, model_type, model_arch, thr=0.5):
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            if model_type == 'torchvision':
                outputs = model(images)['out']
            elif model_type == 'smp':
                outputs = model(images)
            elif model_type == 'huggingface':
                img_processor = AutoImageProcessor.from_pretrained(model_arch)(images = images, return_tensors="pt", do_rescale=False, do_resize=False, do_normalize=False)
                del images
                img_processor['pixel_values'] = img_processor['pixel_values'].half().cuda()
                outputs = model(**img_processor)
                outputs = outputs.logits 
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)    
    avg_dice = torch.mean(dices_per_class).item()
    return avg_dice

def objective(trial, config, CLASSES, CLASS2IND):

    IMAGE_ROOT = config['paths']['train']['image']
    LABEL_ROOT = config['paths']['train']['label']
    DATA_ROOT = config['paths']['data']

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log = True)
    optimizer_type = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'Lion'])
    scheduler_type = trial.suggest_categorical('lr_scheduler', ['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau'])
    scheduler_params = {
        'CosineAnnealingLR':{
            'T_max' : trial.suggest_int('T_max', 10, 50),
            'eta_min': trial.suggest_float('eta_min', 0.0001, 0.001)
        },
        'StepLR':{
            'step_size' : trial.suggest_int('step_size', 5, 20),
            'gamma' : trial.suggest_float('gamma', 0.1, 0.9)
        },
        'ReduceLROnPlateau' : {
            'factor': trial.suggest_float('factor', 0.1, 0.9),
            'patience': trial.suggest_int('patience', 1, 10),
            'mode': 'min'
        }
    }.get(scheduler_type, {})

    pngs, jsons = load_data(IMAGE_ROOT, LABEL_ROOT, DATA_ROOT)

    pngs = sorted(pngs)
    jsons = sorted(jsons)

    tft = None
    tfv = None

    if config['transform']['train'] is not None:
        tft = TransformSelector(config['transform']['train']['type'], config['transform']['train']["augmentations"]).get_transform()
    if config['transform']['val'] is not None:
        tfv = TransformSelector(config['transform']['val']['type'], config['transform']['val']["augmentations"]).get_transform()
    
    train_dataset = XRayDataset(pngs, jsons, DATA_ROOT, CLASSES, CLASS2IND, is_train=True, transforms=tft, debug=config['debug'])
    valid_dataset = XRayDataset(pngs, jsons, DATA_ROOT, CLASSES, CLASS2IND, is_train=False, transforms=tfv, debug=config['debug'])
    
    train_batch = config['training']['batch_size']['train']
    val_batch = config['training']['batch_size']['val']

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=train_batch,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=val_batch,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    model = modelSelector(config['model'], len(CLASSES)).get_model()
    
    if config['optimizer']['type'] == 'Lion':
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    else:
        optimizer = getattr(optim, optimizer_type)(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
    
    scheduler = getattr(optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
    
    criterion = getattr(nn, config['criterion'])()
    
    best_dice = 0.

    scaler = torch.cuda.amp.GradScaler()
    epoch = 50
    model.train()
    for e in tqdm(range(epoch)):
        for step, (images, masks) in enumerate(train_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
                
            with torch.cuda.amp.autocast():
                if config['model']['type'] == 'torchvision':
                    outputs = model(images)['out']
                elif config['model']['type'] == 'smp':
                    outputs = model(images)
                elif config['model']['type'] == 'huggingface':
                    img_processor = AutoImageProcessor.from_pretrained(config['model']['arch'])(images = images, return_tensors="pt", do_rescale=False, do_resize=False, do_normalize=False)
                    del images
                    img_processor['pixel_values'] = img_processor['pixel_values'].half().cuda()
                    outputs = model(**img_processor)
                    outputs = outputs.logits                   

                loss = criterion(outputs, masks)
            optimizer.zero_grad()
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (e + 1) % 10 == 0:
            dice = validation(model, valid_loader, criterion, config['model']['type'], config['model']['arch'])
            best_dice = max(best_dice, dice)

        trial.report(best_dice, e)

        if trial.should_prune():
            raise optuna.TrialPruned()

            # 스케줄러 업데이트
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(best_dice)  # validation metric 기반 업데이트
            else:
                scheduler.step()
    
    #del model
    #gc.collect()
    #torch.cuda.empty_cache()    
    return best_dice  

def main(config, CLASSES, CLASS2IND):
    wandb.init(project='optuna')
    wandb.run.name = config['exp_name']
    wandb.run.save()

    sampler = TPESampler(**TPESampler.hyperopt_parameters())
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(study_name = 'find hyperparameter', direction='maximize', sampler = sampler, pruner = pruner)
    study.optimize(lambda t: objective(t, config, CLASSES, CLASS2IND), n_trials = 3, gc_after_trial=True)
    print(f"Best value: {study.best_value} (params: {study.best_params})")

    SAVED_DIR = os.path.join(config['paths']['model']['save_dir'], config['exp_name'])
    if not os.path.exists(os.path.join(SAVED_DIR)):
        os.mkdir(SAVED_DIR)
    with open(os.path.join(SAVED_DIR, "best_params.json"), "w") as f:
        import json
        json.dump(study.best_params, f, indent=4)