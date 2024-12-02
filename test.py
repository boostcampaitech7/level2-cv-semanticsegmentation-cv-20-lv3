import os
import torch
import albumentations as A
import pandas as pd
from tools.custom_dataset import XRayInferenceDataset
from torch.utils.data import DataLoader
from tools.function import test, tta_func, save_results
from tools.custom_augments import TransformSelector, TestTimeTransform

def main(config, IND2CLASS):
    IMAGE_ROOT = config['paths']['test']['image']
    SAVED_DIR = config['paths']['model']['save_dir']
    thr = 0.5 if not config['pseudo_labeling']['enabled'] else config['pseudo_labeling']['confidence_threshold']

    model = torch.load(config['paths']['model']['pt_loaded_dir'])

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    tfv = TransformSelector(config['transform']['val']['type'], config['transform']['val']["augmentations"]).get_transform()
    test_dataset = XRayInferenceDataset(pngs, IMAGE_ROOT, transforms=tfv)

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    if config['TTA']['used']:
        tta_transform = TestTimeTransform(config['TTA']['augmentations']).getTransform()
        rles, filename_and_class = tta_func(model, tta_transform, IND2CLASS, test_loader, config['model']['type'], config['model']['arch'])
        output = os.path.join(SAVED_DIR, "output")
        save_results(filename_and_class, rles, output,  f"{config['exp_name']}_tta.csv")

    else:
        rles, filename_and_class = test(model, IND2CLASS, test_loader, config['model']['type'], config['model']['arch'], thr=thr)
        output = os.path.join(SAVED_DIR, "output")
        save_results(filename_and_class, rles, output,  f"{config['exp_name']}.csv")

if __name__ == '__main__':
    main()