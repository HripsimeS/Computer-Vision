import logging
import os
import sys
import tempfile
import glob
from monai.transforms import LoadImage
import matplotlib.pyplot as plt
import time
import monai
import nibabel as nib
import numpy as np
import torch
from monai.handlers.utils import from_engine
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image

import argparse
import sys
import os

from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    CropForegroundd,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    EnsureTyped,
)


model_dir  = '../projects/CTPelvic1K_data/train_dir'

parser = argparse.ArgumentParser(description='Predict masks from input images')
parser.add_argument("-i", "--input", type=str, required=True, help="path to input image")
parser.add_argument("-o", "--output", type=str, required=True, help="path to output mask")


def main():
    #print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    args = parser.parse_args()

    test_images = sorted(glob.glob(os.path.join(args.input)))
    test_dicts = [{"image": image_name} for image_name in test_images]
    files = test_dicts[:]



    # define pre transforms
    pre_transforms = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-120, a_max=360,b_min=0.0, b_max=1.0, clip=True,),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys="image"),
        ])
    


    dataset = CacheDataset(data=files, transform=pre_transforms, cache_rate=1.0, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    
    # define post transforms
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            ),
        
        AsDiscreted(keys="pred", argmax=True, to_onehot=5),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out_seg", output_postfix="seg", resample=False),

    ])

    device = torch.device("cuda:0")  
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "3DUnet_best_metric_model.pth")))


    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            start_time = time.time()
            test_inputs = test_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model, sw_device="cuda:0", device="cpu")
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            test_outputs = from_engine(["pred"])(test_data)
            
            loader = LoadImage()
            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])[0]
            test_output_argmax = torch.argmax(test_outputs[0], dim=0, keepdim=True)
            
            rety = blend_images(image=original_image[None], label=test_output_argmax, cmap="jet", alpha=0.5, rescale_arrays=True)
            
            print("Total time seconds: {:.2f}".format((time.time()- start_time)))
    
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image")
            plt.imshow(original_image[None][ 0, :, :, 250], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"predicted mask")
            plt.imshow(test_output_argmax [ 0, :, :, 250],  cmap="jet")
            plt.subplot(1, 3, 3)
            plt.title(f"segmented image")
            plt.imshow(torch.moveaxis(rety[:, :, :, 250], 0, -1))
            plt.show()
            
    
if __name__ == '__main__':
    main()









