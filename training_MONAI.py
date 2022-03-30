#!/usr/bin/env python
# coding: utf-8


import logging
import os
import sys
import tempfile
import glob
import nibabel as nib
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.utils import first, set_determinism
from monai.networks.nets import UNet
import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    Resized,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Activationsd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ScaleIntensityd,
    SaveImaged,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.visualize import plot_2d_or_3d_image
from monai.config import print_config


train_dir  = 'C:/Users/Hripsime/OneDrive - ABYS MEDICAL/projects/CTPelvic1K_data/train_dir'

def main(train_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    train_images = sorted(glob.glob(os.path.join(train_dir, "*data.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(train_dir, "*mask_4label.nii.gz")))
    data_dicts = [{"image": image_name, "mask": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-6], data_dicts[-6:]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-120, a_max=360,b_min=0.0, b_max=1.0, clip=True,),
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
        ),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-120, a_max=360,b_min=0.0, b_max=1.0, clip=True,),
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            EnsureTyped(keys=["image", "mask"]),
        ]
    )

    

    # create a training data loader
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, num_workers=3)
    
    # create a validation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=3)
    
    
   
    
    # create UNet, DiceLoss and Adam optimizer
    device=torch.device("cuda:0")  
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
    optimizer = torch.optim.Adam(model.parameters(), 2e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    VAL_AMP = True
    # define inference method
    def inference(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input, 
                roi_size=(160, 160, 160), 
                sw_batch_size=4, 
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)
    
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    #torch.backends.cudnn.benchmark = True    
    

    # start a typical PyTorch training
    max_epochs = 150
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    
    epoch_loss_values = []
    metric_values = []
    
    writer = SummaryWriter()
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=5)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=5)])
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        

        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["mask"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device), 
                        val_data["mask"].to(device),
                    )
                    val_outputs = inference(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(train_dir, "3DUnet_best_metric_model.pth"))
                    print("saved new best metric model")
                
                print(
                    
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {} Total time seconds: {:.2f}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch, (time.time()- start_time)
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                # plot_2d_or_3d_image(val_inputs, epoch + 1, writer, index=0, tag="image")
                # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="mask")
                # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main(train_dir)







