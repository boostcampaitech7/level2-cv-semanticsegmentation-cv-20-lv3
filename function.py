import os
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F
import datetime
import wandb
import ttach as tta
import torch.nn as nn
import time

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, SAVED_DIR, file_name='best_model.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def set_seed(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def validation(epoch, model, CLASSES, data_loader, criterion, model_type, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            if model_type == 'torchvision':
                outputs = model(images)['out']
            elif model_type == 'smp':
                outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)
            
            dice = dice_coef(outputs, masks)
            wandb.log({'Each dice' : dice})
            dices.append(dice)

    wandb.log({'Validation_loss' : total_loss})

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]

    # 예측 어려운 class 만 wandb로 결과 전송
    target_classes = {"finger-16", "Trapezium", "Trapezoid", "Pisiform", "Lunate"}

    filtered_dice = {
                    f"valid/{entry.split(':')[0].strip()}": float(entry.split(':')[1].strip())  
                    for entry in dice_str if entry.split(":")[0].strip() in target_classes
                    }
    wandb.log(filtered_dice)
    
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    print(f'avg_dice: {avg_dice}')
    return avg_dice

def train(model, NUM_EPOCHS, CLASSES, train_loader, val_loader, criterion, optimizer, VAL_EVERY, SAVED_DIR, model_name, model_type, patience, delta, scheduler=None):
    
    print(f'Start training..')
    
    best_dice = 0.
    # patience = 10  # EarlyStopping patience
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta = delta)
    scaler = torch.cuda.amp.GradScaler()
  
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()  # 에포크 시작 시간 기록

        model.train()
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        # 에포크 시작 시 로깅
        wandb.log({'Epoch': epoch + 1})
        
        for step, (images, masks) in enumerate(train_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            with torch.cuda.amp.autocast():
                if model_type == 'torchvision':
                    outputs = model(images)['out']
                elif model_type == 'smp':
                    outputs = model(images)

                loss = criterion(outputs, masks)
            optimizer.zero_grad()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                wandb.log({'Train_loss' : loss.item()})

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] completed in {epoch_time:.2f} seconds.')    

        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, CLASSES, val_loader, criterion, model_type)
            wandb.log({'Average_dice': dice})

            early_stopping(dice, model, epoch + 1)

            if early_stopping.verbose:
                print(f"Current best score: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")

            if early_stopping.early_stop:
                if dice > early_stopping.best_epoch:
                    save_model(model, SAVED_DIR, f"{model_name}_best_{epoch + 1}_epoch.pt")
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best model saved at epoch {epoch + 1}.")

                else:    
                    save_model(early_stopping.best_model_state,SAVED_DIR,f"{model_name}_best_{early_stopping.best_epoch}_epoch.pt"            )
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best model saved at epoch {early_stopping.best_epoch}.")
                break

        # 스케줄러 업데이트
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(best_dice)  # validation metric 기반 업데이트
            else:
                scheduler.step()
            wandb.log({'Learning_rate': optimizer.param_groups[0]['lr']})
 
    if not early_stopping.early_stop:
        save_model(model, SAVED_DIR, f'{model_name}_last_epoch.pt')
        print(f'Training completed. Final model saved as {model_name}_last_epoch.pt')


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다.

def decode_rle_to_mask(rle, height, width):

    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

def test(model, IND2CLASS, data_loader, model_type, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            if model_type == 'torchvision':
                outputs = model(images)['out']
            elif model_type == 'smp':
                outputs = model(images)
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.01):
        """
        Args:
            patience (int): 개선되지 않아도 기다리는 최대 에포크 수
            verbose (bool): 개선될 때마다 로그를 출력할지 여부
            delta (float): 개선된 것으로 간주하기 위한 최소 변화량
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_epoch = -1
        self.best_model_state = None

    def __call__(self, val_score, model, epoch):
        score = val_score

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.best_model_state = model.state_dict()  # Best 모델 상태 저장
            if self.verbose:
                print(f"Improved score to {score:.4f} at epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def tta_func(model, tta_transforms, IND2CLASS, data_loader, model_type, thr=0.5):

    if model_type == 'torchvision':
        class CustomSegmentationModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                outputs = self.model(x)
                return outputs['out']

        model = CustomSegmentationModel(model)
    
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()

            tta_model = tta.SegmentationTTAWrapper(model, tta_transforms)
            outputs = tta_model(images)
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class
