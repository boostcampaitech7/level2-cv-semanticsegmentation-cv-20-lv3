from torchvision import models
import torch.nn as nn
import segmentation_models_pytorch as smp

class torchvisionModel:
    def __init__(self, model_config):
        self.model = getattr(models.segmentation, model_config['name'])(pretrained = model_config['pretrained'])
    
    def get_model(self):
        return self.model

class smpModel:
    def __init__(self, model_config, num_classes):
        self.arch = model_config['arch']
        self.encoder_name = model_config['encoder_name']
        self.encoder_weights = model_config['encoder_weights']
        self.in_channels = model_config['in_channels']
        self.num_classes = num_classes

    def get_model(self):
        self.model = smp.create_model(
            arch = self.arch,
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = self.in_channels,
            classes = self.num_classes
        )
        return self.model 

class modelSelector:
    def __init__(self, model_config, num_classes):
        if not model_config['type'] in ["torchvision", "smp"]:
            raise ValueError("Unknown model library specified.")
        self.model_config = model_config
        self.num_classes = num_classes

    def get_model(self):
        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.model_config['type'] == 'torchvision':
            model = torchvisionModel(self.model_config).get_model()
            model = changeModule(model, self.model_config['name'], self.num_classes)
        elif self.model_config['type'] == 'smp':
            model = smpModel(self.model_config, self.num_classes).get_model()

        return model

def changeModule(model, model_name, num_classes):
    if "fcn" in model_name:
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    if "deeplabv3" in model_name:
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if "lraspp" in model_name:
        model.classifier.low_classifier = nn.Conv2d(
            model.classifier.low_classifier.in_channels, num_classes, kernel_size=1
        )
        model.classifier.high_classifier = nn.Conv2d(
            model.classifier.high_classifier.in_channels, num_classes, kernel_size=1
        )
    return model