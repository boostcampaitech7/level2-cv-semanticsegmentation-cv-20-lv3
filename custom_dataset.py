import os
import json
import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(self, pngs, jsons, img_root, label_root, classes, CLASS2IND, is_train=True, transforms=None, debug=False):
        self.img_root = img_root
        self.label_root = label_root
        self.classes = classes
        self.CLASS2IND = CLASS2IND

        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)

        # debug 모드일 때는 전체 데이터를 사용하지 않고 5%만 샘플링합니다.
        if debug:
            debug_sample_size = int(len(_filenames) * 0.05)
            indices = np.random.choice(len(_filenames), debug_sample_size, replace=False)
            self.filenames = _filenames[indices].tolist()
            self.labelnames = _labelnames[indices].tolist()
            print(f"Debug mode enabled: Using {debug_sample_size} samples")
        else:
            # split train-valid
            groups = [os.path.dirname(fname) for fname in _filenames]
            ys = [0 for fname in _filenames]
            gkf = GroupKFold(n_splits=5)
            
            filenames = []
            labelnames = []
            for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
                if is_train:
                    if i == 0:
                        continue  # 첫 번째 fold를 validation으로 사용
                    
                    filenames += list(_filenames[y])
                    labelnames += list(_labelnames[y])
                
                else:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])
                    break  # skip i > 0
            
            self.filenames = filenames
            self.labelnames = labelnames
            self.is_train = is_train
            self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.img_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(self.classes), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = self.CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # channel first 포맷으로 변경합니다.
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label


class XRayInferenceDataset(Dataset):
    def __init__(self, pngs, img_root, transforms=None):
        self.img_root = img_root
        
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.img_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name