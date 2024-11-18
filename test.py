import os
import torch
import albumentations as A
import pandas as pd
from custom_dataset import XRayInferenceDataset
from torch.utils.data import DataLoader
from function import test
from custom_augments import TransformSelector

def main(config, IND2CLASS):
    IMAGE_ROOT = config['test_img']
    SAVED_DIR = config['pt_saved_dir']

    model = torch.load(config['pt_loaded_dir'])

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    tf = TransformSelector(config['transform']['transform_type'], config['transform']["augmentations"]).get_transform()
    test_dataset = XRayInferenceDataset(pngs, IMAGE_ROOT, transforms=tf)

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    rles, filename_and_class = test(model, IND2CLASS, test_loader, config['model']['type'])

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    output = os.path.join(SAVED_DIR, "output")
    if not os.path.exists(output):
        os.makedirs(output)
    df.to_csv(os.path.join(output, f"{config['exp_name']}.csv"), index=False)

if __name__ == '__main__':
    main()