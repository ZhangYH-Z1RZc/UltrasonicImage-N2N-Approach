import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path
from runpy import run_path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from tqdm import tqdm

import config
import wandb
from denoising_diffusion_pytorch import Diffusion_Unet, GaussianDiffusion
from nets import CostumeUNet, N2N_Original_Used_UNet, ClassicalUnet, MultiResUnet
from utils.dataset_creator import CreateDataset_ForNoiseRGBToNoiseRGB_Training, \
    CreateDataset_ForNoiseRGBToBWMask_Training, CreateDataset_ForNoiseRGBToCleanRGB_Training

torch.autograd.set_detect_anomaly(True)
import segmentation_models_pytorch as smp


def Get_Dataset(params):
    """Return the suitable dataset base on the model output features"""
    log.info(f"Output config is {config.final_channel} output channels.")
    if config.final_channel == 3:
        if config.n2n_mode:
            log.info("Creating Dataset for N2N Training")
            result = CreateDataset_ForNoiseRGBToNoiseRGB_Training(params["dataPrefix"])()
        elif config.n2c_mode:
            log.info("Creating Dataset for N2C Training")
            result = CreateDataset_ForNoiseRGBToCleanRGB_Training(params["dataPrefix"])()
        else:
            log.info("No proper training mode assigned, using N2N mode")
            result = CreateDataset_ForNoiseRGBToNoiseRGB_Training(params["dataPrefix"])()
        """This syntax means First initialize the object then call it as a callable."""
    elif config.final_channel == 1:
        result = CreateDataset_ForNoiseRGBToBWMask_Training(params["dataPrefix"])()
    else:
        raise Exception("The Network output Channel(s) doesn't match any Implementation")
    log.info("Dataset Created")
    return result


def Check_ImageChannel(params):
    """Make sure the target has same amount of channels as the last layer of network has"""
    for batch in DataLoader(dataset=Get_Dataset(params), shuffle=True,
                            batch_size=params["batch_size"], num_workers=4, pin_memory=True):
        images = batch[0]
        images = images.to(device=config.device)
        assert images.shape[1] == params["net"].in_channels, (
                f'Network has been defined with {params["net"].in_channels} input channels, ' +
                f'but loaded images have {images.shape[1]} channels. Please check that '
                'the images are loaded correctly.')


def Move_Batch_To_Device(batch):
    images = batch[0]
    images = images.to(device=config.device)
    true_masks = batch[1]
    true_masks = true_masks.to(device=config.device)
    return images, true_masks


def Save_CheckPoint(global_step, params):
    """Save the current param dict base on step (every 1000 step)"""
    if global_step % 1000 == 0:
        torch.save(params["net"].state_dict(), 'INTERRUPTED.pth')
        log.info('Saved interrupt')


def Update_Progress_Bar(pbar, images, loss_item):
    pbar.update(images.shape[0])
    pbar.set_postfix(**{'loss (batch)': loss_item})


def Wandb_LogTrainning(loss_item, global_step, epoch):
    """Log WanDB at the end of a each batch"""
    wandb_experiment.log({
        'train loss': loss_item,
        'step': global_step,
        'epoch': epoch
    })


def Compute_Loss_do_backward(params, aOptimizer, aGradScaler, someImages, someTrue_masks):
    with torch.cuda.amp.autocast(enabled=params["amp"]):
        # log.info("Tring to enable torch.cuda.amp.autocast")
        if config.using_fcn_50 or config.using_fcn_101:
            masks_pred = params["net"](someImages)['out']
        else:
            masks_pred = params["net"](someImages)
        """Explain ['out']
        The FCN_ResNet model returns an OrderedDict with two Tensors that are of the same height and width as the input Tensor, 
        but with 21 classes. output['out'] contains the semantic masks, 
        and output['aux'] contains the auxillary loss values per-pixel. 
        In inference mode, output['aux'] is not useful. So, output['out'] is of shape (N, 21, H, W)."""
        if config.using_L1loss:
            # log.debug("Using L1 for loss computation")
            l1 = nn.L1Loss()
            loss = l1(masks_pred, someTrue_masks)
        elif config.using_MSELoss:
            # log.info("Using L2 (MSE) for loss computation")
            mse = nn.MSELoss(reduction='sum')
            loss = mse(masks_pred, someTrue_masks)
        elif config.using_CrossEntropyLoss:
            # log.info("Using CrossEntropyLoss")
            cel = nn.CrossEntropyLoss()
            loss = cel(masks_pred, someTrue_masks)
        elif config.using_HuberLoss:
            # log.info("Using HuberLoss ")
            hl = nn.HuberLoss()
            loss = hl(masks_pred, someTrue_masks)
        elif config.using_SmoothL1Loss:
            # log.info("Using SmoothL1Loss")
            sl1 = nn.SmoothL1Loss()
            loss = sl1(masks_pred, someTrue_masks)
        elif config.using_combined_L1_MSE:
            # log.info("Using COmbined lOss L1+MSE")
            l1 = nn.L1Loss()
            mse = nn.MSELoss(reduction='sum')
            loss_1 = l1(masks_pred, someTrue_masks)
            loss_2 = mse(masks_pred, someTrue_masks)
            loss = torch.add(loss_2, loss_1)
        elif config.using_combined_SL1_MSE:
            # log.info("Using COmbined lOss SL1+MSE")
            sl1 = nn.SmoothL1Loss()
            mse = nn.MSELoss(reduction='sum')
            loss_1 = sl1(masks_pred, someTrue_masks)
            loss_2 = mse(masks_pred, someTrue_masks)
            loss = torch.add(loss_2, loss_1)
        elif config.using_combined_huber_MSE:
            # log.info("Using COmbined lOss Huber+MSE")
            huber = nn.HuberLoss()
            mse = nn.MSELoss(reduction='sum')
            loss_1 = huber(masks_pred, someTrue_masks)
            loss_2 = mse(masks_pred, someTrue_masks)
            loss = torch.add(loss_2, loss_1)
        elif config.using_combined_all_loss:
            # log.info("Using Combiend lOss All")
            huber = nn.HuberLoss()
            sl1 = nn.SmoothL1Loss()
            l1 = nn.L1Loss()
            mse = nn.MSELoss(reduction='sum')
            loss_1 = huber(masks_pred, someTrue_masks)
            loss_2 = mse(masks_pred, someTrue_masks)
            loss_3 = sl1(masks_pred, someTrue_masks)
            loss_4 = l1(masks_pred, someTrue_masks)
            loss = loss_1 + loss_4 + loss_3 + loss_2
    aOptimizer.zero_grad(set_to_none=True)
    aGradScaler.scale(loss).backward()
    aGradScaler.step(aOptimizer)
    aGradScaler.update()
    return loss.item()


def Compute_GaussianDiffusion_loss_do_backward(params, aOptimizer, aGrad_scaler,
                                               someTargetDistributionImage, someConditionImages):
    loss = params["diffusion_model"](img=someTargetDistributionImage, condition_img=someConditionImages)
    aOptimizer.zero_grad(set_to_none=True)
    aGrad_scaler.scale(loss).backward()
    aGrad_scaler.step(aOptimizer)
    aGrad_scaler.update()

    return loss.item()


def Train_In_Batch(params, aOptimizer, aGrad_scaler, pbar, epoch):
    """Train the network in batch"""
    # Check_ImageChannel(params)
    # this is too time consuming
    global_step = 0
    for batch in DataLoader(dataset=Get_Dataset(params), shuffle=True,
                            batch_size=params["batch_size"], num_workers=4, pin_memory=True):
        images, true_masks = Move_Batch_To_Device(batch)
        if config.using_gaussian_diffusion:
            loss_item = Compute_GaussianDiffusion_loss_do_backward(params, aOptimizer, aGrad_scaler,
                                                                   someTargetDistributionImage=true_masks,
                                                                   someConditionImages=images)
        else:
            loss_item = Compute_Loss_do_backward(params, aOptimizer, aGrad_scaler, images, true_masks)
        Save_CheckPoint(global_step, params)

        """Each batch one increment in step"""
        global_step += 1
        Wandb_LogTrainning(loss_item, global_step, epoch)
        Update_Progress_Bar(pbar, images, loss_item)


def Save_CheckPoint_Each_Epoch(params, epoch):
    """Save CheckPoint in the end of each epoch"""
    if params["save_checkpoint"]:
        dir_checkpoint = "checkpoints"
        today = date.today()
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        if config.using_gaussian_diffusion:
            torch.save(params["diffusion_model"].state_dict(),
                       f"{dir_checkpoint}/{today}_checkpoint_{params['dataPrefix']}_epoch_{epoch + 1}.pth")
        else:
            torch.save(params["net"].state_dict(),
                       f"{dir_checkpoint}/{today}_checkpoint_{params['dataPrefix']}_epoch_{epoch + 1}.pth")
        log.info(f'Checkpoint for {params["dataPrefix"]} {epoch + 1} saved!')


def Train_Loop(params, aOptimizer, aGrad_scaler):
    # 5. Begin training
    for epoch in range(params["epochs"]):
        params["net"].train()

        with tqdm(total=len(Get_Dataset(params)), desc=f'Epoch {epoch + 1}/{params["epochs"]}', unit='img') as pbar:
            Train_In_Batch(params, aOptimizer, aGrad_scaler, pbar, epoch)
        Save_CheckPoint_Each_Epoch(params, epoch)


def train_net(params):
    """The main entry for training"""
    # Update wandb logging
    wandb_experiment.config.update(dict(epochs=params["epochs"],
                                        batch_size=params["batch_size"],
                                        learning_rate=params["learning_rate"],
                                        save_checkpoint=params["save_checkpoint"],
                                        amp=params["amp"]))

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if config.using_gaussian_diffusion:
        log.info("Setting Up Optimizer for diffusion Model")
        optimizer = optim.RMSprop(params["diffusion_model"].parameters(), lr=params["learning_rate"], weight_decay=1e-8,
                                  momentum=0.9)
    else:
        optimizer = optim.RMSprop(params["net"].parameters(), lr=params["learning_rate"], weight_decay=1e-8,
                                  momentum=0.9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=params["amp"])

    Train_Loop(params, optimizer, grad_scaler)


def get_args():
    """Return in dict form"""
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--run-name', '-rn', dest='run_name', type=str, default="Run Name", help='Run Name For Wandb')
    parser.add_argument('--project-name', '-pn', dest='project_name',
                        type=str, default="Project Name", help='Run Name For Wandb')
    parser.add_argument('--prefix', '-p', dest='dataPrefix', default="Bodymark_Dataset", help='dataset folder prefix')
    return vars(parser.parse_args())


def Get_Model(args):
    log.info(f'Using device {config.device}')
    if config.using_fcn_50:
        log.info("Using FCN_ResNet model for training")
        result = fcn_resnet50(pretrained_backbone=False, num_classes=3)
    elif config.using_fcn_101:
        log.info("Using FCN_ResNet model for testing, postProcess utility switch to fcn mode")
        result = fcn_resnet101(pretrained_backbone=False, num_classes=3)
    elif config.using_gaussian_diffusion:
        log.info("Using Gaussiance diffusion model")
        result = Diffusion_Unet(dim=16, dim_mults=(1, 2, 4, 8),
                                channels=config.final_channel,
                                condition_channels=config.image_data_channel)
    elif config.using_multiresunet:
        log.info("Using Multires-Unet.")
        result = MultiResUnet(channels=3, nclasses=config.final_channel)
    elif config.using_N2N_original_unet:
        log.info("Using U-Net Used in first Noise2noise paper")
        result = N2N_Original_Used_UNet(in_channels=3, out_channels=config.final_channel)
    elif config.using_ClassicalUnet:
        log.info("Using U-Net used in first U-net Paper")
        result = ClassicalUnet(n_channels=3, n_classes=config.final_channel)
    elif config.using_Costume_Unet or config.using_Costume_Unet_no_StdNorm:
        log.info("Using Our Costume U-Net")
        result = CostumeUNet(in_channels=3, out_channels=config.final_channel)
    elif config.using_UNet_pp:
        log.info("Using External Unet++")
        result = smp.UnetPlusPlus(in_channels=3, classes=config.final_channel, encoder_weights=None)
    elif config.using_DeepLabV3:
        log.info("Using External DeepLabV3")
        result = smp.DeepLabV3(in_channels=3, classes=config.final_channel, encoder_weights=None, activation=None)
    elif config.using_Linknet:
        log.info("Using External Linknet")
        result = smp.Linknet(in_channels=3, classes=config.final_channel, encoder_weights=None, activation=None)
    elif config.using_MAnet:
        log.info("Using External MAnet")
        result = smp.MAnet(in_channels=3, classes=config.final_channel, encoder_weights=None, activation=None)
    elif config.using_restormer:
        log.info("Using Restormer")
        parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 6, 'num_blocks': [1, 1, 1, 2],
                      'num_refinement_blocks': 2, 'heads': [1, 1, 2, 2], 'ffn_expansion_factor': 1.02, 'bias': False,
                      'LayerNorm_type': 'WithBias', 'dual_pixel_task': False}
        load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
        result = load_arch['Restormer'](**parameters)
    else:
        log.info("No Model Specification provided, using Cosutume U-Net")
        result = CostumeUNet(in_channels=3, out_channels=config.final_channel)
    log.info("Model Loaded")
    if args["load"]:
        result.load_state_dict(torch.load(args["load"], map_location=config.device))
        log.info(f'Model loaded from {args["load"]}')

    result.to(device=config.device)
    return result


def main(args):
    Main_Net = Get_Model(args)
    train_parameters = {"net": Main_Net,
                        "epochs": args["epochs"],
                        "dataPrefix": args["dataPrefix"],
                        "save_checkpoint": config.save_checkpoint,
                        "batch_size": config.batch_size,
                        "learning_rate": config.learning_rate,
                        "device": config.device,
                        "amp": config.amp,
                        "data_root": config.data_root
                        }
    if config.using_gaussian_diffusion:
        log.info("Createing Diffusion Model")
        diffusion = GaussianDiffusion(
            model=train_parameters["net"],
            image_size=config.classic_input_size[0],
            timesteps=1000,
            loss_type="l1"
        )
        log.info("Adding keyword 'diffusion_model' to train_parameters")
        train_parameters["diffusion_model"] = diffusion
    try:
        train_net(params=train_parameters)
    except KeyboardInterrupt:
        torch.save(Main_Net.state_dict(), 'INTERRUPTED.pth')
        log.info('Saved interrupt')
        sys.exit(0)


if __name__ == '__main__':
    # TODO: sync all args in form of dict, resolve confliction
    args = get_args()
    wandb_experiment = wandb.init(project=args['project_name'], resume='allow')
    wandb.run.name = f"{args['run_name']} with{args['dataPrefix']}"
    log = logging.getLogger(__name__)
    main(args)
