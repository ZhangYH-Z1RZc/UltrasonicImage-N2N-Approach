import argparse
import utils.postProcessUtils as post
import logging
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import label, generate_binary_structure
from torch.utils.data import DataLoader
from nets import CostumeUNet, N2N_Original_Used_UNet, ClassicalUnet, MultiResUnet
from utils.n2ndataloader import TestDataFeeder
import config
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from denoising_diffusion_pytorch import Diffusion_Unet, GaussianDiffusion
import segmentation_models_pytorch as smp
import timm
from runpy import run_path
import os

log = logging.getLogger(__name__)


def Score_Computation_Loop(net, source_data_Loader, params):
    Dice_values = []
    IoU_values = []
    AP_values = []
    SSIM_value = []
    PSNR_value = []
    PSNR_HVS_M_value = []
    for step, (source, truth_mask, clean_original) in enumerate(source_data_Loader):
        step_processor = post.BodyMarkPostProcessor(source, truth_mask, clean_original,net,
                                                    output_folderPrefix=params['test_export_location_prefix'],
                                                    step=step)

        """There is no transform for target_transformation, so truth mask will be in the
        form of PIL image, but source is a normalized tensor"""

        Dice_values.append(step_processor.Compute_Dice_Score())
        # log.info(f"Dice Score: {step_processor.Compute_Dice_Score()}")
        step_processor.Compute_Score_Average_Every_X_Step(Dice_values, "Dice")

        IoU_values.append(step_processor.Compute_IoU_Score())
        # log.info(f"IoU Score: {step_processor.Compute_IoU_Score()}")
        step_processor.Compute_Score_Average_Every_X_Step(IoU_values, "IoU")

        AP_values.append(step_processor.Compute_PA_Score())
        # log.info(f"PA Score: {step_processor.Compute_PA_Score()}")
        step_processor.Compute_Score_Average_Every_X_Step(AP_values, "AP")

        SSIM_value.append(step_processor.Compute_SSIM_Score())
        step_processor.Compute_Score_Average_Every_X_Step(SSIM_value, "SSIM")

        PSNR_value.append(step_processor.Compute_PSNR_Score())
        step_processor.Compute_Score_Average_Every_X_Step(PSNR_value, "PSNR")

        PSNR_HVS_M_value.append(step_processor.Compute_PSNR_HVS_M_Score())
        step_processor.Compute_Score_Average_Every_X_Step(PSNR_HVS_M_value,"PSNR_HSV_M")
        # log.info(np.array(PSNR_HVS_M_value).mean())
        # log.info(np.array(PSNR_HVS_M_value).max())
        # log.info(np.array(PSNR_HVS_M_value).min())
        step_processor.Write_File_Of_Interest()
        step_processor.Write_File_every_10_Step()


def get_args():
    parser = argparse.ArgumentParser(description='Run Test On model')
    parser.add_argument('--data-root', '-d', dest='data_root',
                        type=str, default="data", help='Root location of ')
    parser.add_argument('--noised-prefix', '-np', dest='test_noised_data_prefix',
                        type=str, default="Bodymark_Dataset",
                        help='path to folder in data root, where img with noised is stored')
    parser.add_argument('--export-prefix', '-ep', dest='test_export_location_prefix',
                        type=str, default="Bodymark_Dataset_PostProcessEXP2",
                        help='where to export results')
    parser.add_argument('--checkpoint', '-c', dest='checkpoint',
                        type=str,
                        default="checkpoints/BodyMark/N2N_Linknet_N2C/2022-12-07_checkpoint_Bodymark_Dataset_epoch_10.pth",
                        help='path to checkpoint')
    return vars(parser.parse_args())


def Construct_Dataset(dataset_folderPrefix: str = "Test"):
    dir_img = f"{config.data_root}/{dataset_folderPrefix}"
    if config.final_channel == 1:
        source_data = TestDataFeeder(sourceDir=dir_img,
                                     transform=config.dataset_input_color_image_2_tensor_transform,
                                     target_transform=config.transform_output_bw_image_2_tensor_transform)  # Create DataSet
    if config.final_channel == 3:
        source_data = TestDataFeeder(sourceDir=dir_img,
                                     transform=config.dataset_input_color_image_2_tensor_transform,
                                     target_transform=config.dataset_input_color_image_2_tensor_transform)  # Create DataSet
    source_data_Loader = DataLoader(source_data, batch_size=config.test_mode_batch_size, shuffle=False)
    return source_data_Loader


def Get_Model(args):
    if config.using_fcn_50:
        log.info("Using FCN_ResNet model for testing, postProcess utility switch to fcn mode")
        result = fcn_resnet50(pretrained_backbone=False, num_classes=3)
    elif config.using_fcn_101:
        log.info("Using FCN_ResNet model for testing, postProcess utility switch to fcn mode")
        result = fcn_resnet101(pretrained_backbone=False, num_classes=3)
    elif config.using_gaussian_diffusion:
        log.info("Using Gaussian Diffusion Model For Testing")
        diffusion_main_frame_network = Diffusion_Unet(dim=16, dim_mults=(1, 2, 4, 8),
                                            channels=config.final_channel,
                                            condition_channels=config.image_data_channel)
        result = GaussianDiffusion(
            model=diffusion_main_frame_network,
            image_size=config.classic_input_size[0],
            timesteps=1000,
            loss_type="l1"
        )
    elif config.using_multiresunet:
        log.info("Using MultiRes-Unet for Testing")
        result = MultiResUnet(channels=config.image_data_channel, nclasses=config.final_channel)
    elif config.using_N2N_original_unet:
        log.info("Using U-Net Used in first Noise2noise paper for Testing")
        result = N2N_Original_Used_UNet(in_channels=3, out_channels=config.final_channel)
    elif config.using_ClassicalUnet:
        log.info("Using U-Net used in first U-net Paper for Testing")
        result = ClassicalUnet(n_channels=3, n_classes=config.final_channel)
    elif config.using_Costume_Unet:
        log.info("Using Our Costume U-Net for Testing")
        result = CostumeUNet(in_channels=3, out_channels=config.final_channel)
    elif config.using_UNet_pp:
        log.info("Using External Unet++")
        result = smp.UnetPlusPlus(in_channels=3, classes=config.final_channel, encoder_weights=None, activation=None)
    elif config.using_DeepLabV3:
        log.info("Using External DeepLabV3 for Testing")
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
        log.info("No specification detected, using Our Costume U-Net for Testing")
        result = CostumeUNet(in_channels=config.image_data_channel, out_channels=config.final_channel)
    result.load_state_dict(torch.load(args["checkpoint"], map_location=config.device))
    result = result.to(device=config.device)
    result.eval()
    return result


def main(param):
    source_data_Loader = Construct_Dataset(dataset_folderPrefix=param['test_noised_data_prefix'])
    net = Get_Model(param)
    Score_Computation_Loop(net, source_data_Loader, param)


if __name__ == "__main__":
    args = get_args()
    main(args)
