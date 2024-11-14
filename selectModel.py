from torchvision import models
import torch.nn as nn

class torchvisionModel:
    def __init__(self, model_config):
        self.model = getattr(models.segmentation, model_config['name'])(pretrained = model_config['pretrained'])
    
    def get_model(self):
        return self.model

class modelSelector:
    def __init__(self, model_config, num_classes):
        if not model_config['type'] in ["torchvision"]:
            raise ValueError("Unknown model library specified.")
        self.model_config = model_config
        self.num_classes = num_classes
    def get_model(self):
        
        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.model_config['type'] == 'torchvision':
            model = torchvisionModel(self.model_config).get_model()
        return changeModule(model, self.model_config['name'], self.num_classes) 
    
def changeModule(model, model_name, num_classes):
    if "fcn" in model_name:
       model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1) 
    return model