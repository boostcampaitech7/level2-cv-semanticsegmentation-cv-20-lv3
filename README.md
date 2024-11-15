### Config file format
Config files are in `.yaml` format:
```yaml
proj_name : baseline
exp_name : basetest

random_seed: 20

model:
  type: torchvision
  name: fcn_resnet50
  pretrained: True

train_img: '/data/ephemeral/home/data/train/DCM'
train_label: '/data/ephemeral/home/data/train/outputs_json'
test_img: /data/ephemeral/home/data/test/DCM
pt_saved_dir: '/data/ephemeral/home/lsh/result'
pt_loaded_dir: '/data/ephemeral/home/lsh/result/basetest/fcn_resnet50.pt'

transform:
  transform_type: albumentations
  augmentations:
    - type: Resize
      params:
        height: 512
        width: 512

train_batch: 4
val_batch: 8
num_epochs: 200
val_every: 5
lr: 0.0001

criterion: BCEWithLogitsLoss
optimizer: Adam

```

Add addional configurations if you need.

### Training with config example
Modify the configurations in `.yaml` config files, then run:

  ```
  python seg.py --mode train --config ./config/config.yaml
  ```


### Test with config example
Modify the configurations in `.yaml` config files, then run:

  ```
  python seg.py --mode test --config ./config/config.yaml
  ```
