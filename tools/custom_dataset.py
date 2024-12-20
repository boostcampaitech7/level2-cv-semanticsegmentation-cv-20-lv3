import os
import json
import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
import os.path as osp

class XRayDataset(Dataset):
    def __init__(self, pngs, jsons, cropped_pngs, cropped_jsons, data_root, classes, CLASS2IND, is_train=True, transforms=None, debug=False, cropped = False):
        self.data_root = data_root
        self.classes = classes
        self.CLASS2IND = CLASS2IND

        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        _cropped_filenames = np.array(cropped_pngs)
        _cropped_labelnames = np.array(cropped_jsons)

        # debug 모드일 때는 전체 데이터를 사용하지 않고 5%만 샘플링합니다.
        if debug:
            debug_sample_size = int(len(_filenames) * 0.05)
            indices = np.random.choice(len(_filenames), debug_sample_size, replace=False)
            self.filenames = _filenames[indices].tolist()
            self.labelnames = _labelnames[indices].tolist()
            self.is_train = is_train
            self.transforms = transforms
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
            if cropped:
                if is_train:
                    filenames += list(_cropped_filenames)
                    labelnames += list(_cropped_labelnames)
            else:
                pass
            self.filenames = filenames
            self.labelnames = labelnames
            self.is_train = is_train
            self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):

        image_name = self.filenames[item]
        image_path = os.path.join(self.data_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.data_root, label_name)
        
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

            # pseudo labeling이므로, 예측이 아예 안 된 라벨이 존재할 수 있다.
            if points.size: cv2.fillPoly(class_label, [points], 1)
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
    
class EnsembleDataset(Dataset):
    """
    Soft Voting을 위한 DataSet 클래스입니다. 이 클래스는 이미지를 로드하고 전처리하는 작업과
    구성 파일에서 지정된 변환을 적용하는 역할을 수행합니다.

    Args:
        fnames (set) : 로드할 이미지 파일 이름들의 set
        cfg (dict) : 이미지 루트 및 클래스 레이블 등 설정을 포함한 구성 객체
        tf_dict (dict) : 이미지에 적용할 Resize 변환들의 dict
    """    
    def __init__(self, fnames, cfg, tf_dict):
        self.fnames = np.array(sorted(fnames))
        self.image_root = cfg.image_root
        self.tf_dict = tf_dict
        self.ind2class = {i : v for i, v in enumerate(cfg.CLASSES)}

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, item):
        """
        지정된 인덱스에 해당하는 이미지를 로드하여 반환합니다.
        Args:
            item (int): 로드할 이미지의 index

        Returns:
            dict : "image", "image_name"을 키값으로 가지는 dict
        """        
        image_name = self.fnames[item]
        image_path = osp.join(self.image_root, image_name)
        image = cv2.imread(image_path)

        assert image is not None, f"{image_path} 해당 이미지를 찾지 못했습니다."
        
        image = image / 255.0
        return {"image" : image, "image_name" : image_name}

    def collate_fn(self, batch):
        """
        배치 데이터를 처리하는 커스텀 collate 함수입니다.

        Args:
            batch (list): __getitem__에서 반환된 데이터들의 list

        Returns:
            dict: 처리된 이미지들을 가지는 dict
            list: 이미지 이름으로 구성된 list
        """        
        images = [data['image'] for data in batch]
        image_names = [data['image_name'] for data in batch]
        inputs = {"images" : images}

        image_dict = self._apply_transforms(inputs)

        image_dict = {k : torch.from_numpy(v.transpose(0, 3, 1, 2)).float()
                      for k, v in image_dict.items()}
        
        for image_size, image_batch in image_dict.items():
            assert len(image_batch.shape) == 4, \
                f"collate_fn 내부에서 image_batch의 차원은 반드시 4차원이어야 합니다.\n \
                현재 shape : {image_batch.shape}"
            assert image_batch.shape == (len(batch), 3, image_size, image_size), \
                f"collate_fn 내부에서 image_batch의 shape은 ({len(batch)}, 3, {image_size}, {image_size})이어야 합니다.\n \
                현재 shape : {image_batch.shape}"

        return image_dict, image_names
    
    def _apply_transforms(self, inputs):
        """
        입력된 이미지에 변환을 적용합니다.

        Args:
            inputs (dict): 변환할 이미지를 포함하는 딕셔너리

        Returns:
            dict : 변환된 이미지들
        """        
        return {
            key: np.array(pipeline(**inputs)['images']) for key, pipeline in self.tf_dict.items()
        }
