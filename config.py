"""This file contains all the configuration that is not exposed to arg"""

import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

import torch
import torchvision.transforms as T

# 0. Config Main Switch
config_on = True

# 1. logging config
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

console_handler = StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

file_handler = RotatingFileHandler("train.log", "a", maxBytes=1024 * 1024 * 5, backupCount=1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

root_logger = logging.getLogger()
# root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.DEBUG)

# 3. Pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4. Training Hyper Parameters
learning_rate = 0.00001
batch_size = 4
test_mode_batch_size = 1
amp = False
save_checkpoint = True

# 5. Model Structure Hyper Parameters
image_data_channel = 3
final_channel = 3

using_fcn_50 = 0
using_fcn_101 = 0
using_gaussian_diffusion = 0
using_multiresunet = 0
using_N2N_original_unet = 0
using_ClassicalUnet = 0
using_Costume_Unet = 0
using_UNet_pp = 0
using_DeepLabV3 = 1
using_Linknet = 0
using_MAnet = 0
using_Costume_Unet_no_StdNorm = 0
using_restormer = 0
# 6. Training Mode
n2n_mode = 1
n2c_mode = 0
n2bw_mode = 0

# 7. Loss Function Switch
using_L1loss = 0
using_MSELoss = 0
using_CrossEntropyLoss = 0
using_HuberLoss = 0
using_SmoothL1Loss = 1
using_combined_L1_MSE = 0
using_combined_SL1_MSE = 0
using_combined_huber_MSE = 0
using_combined_all_loss = 0

# 8. PostProcess COnfig
using_conectivity_filter = 1

# 2. dataset config
dataset_mean = [46.166 / 255, 45.238 / 255, 44.75711 / 255]
dataset_std = [59.3565 / 255, 59.473 / 255, 59.923 / 255]
dataset_inverse_mean = [-m / s for m, s in zip(dataset_mean, dataset_std)]
dataset_inverse_std = [1 / s for s in dataset_std]

classic_input_size = (512, 512)  # 800 x 800
classic_output_size = (388, 388)  # 800 x 800

costume_input_size = (800, 800)
costume_output_size = (800, 800)

"""Disabling All sub normalization, using ToTensor()'s normalization"""
if using_gaussian_diffusion or using_UNet_pp or using_DeepLabV3 or using_Costume_Unet_no_StdNorm:
    root_logger.info("Using Transform without T.Normalize")
    dataset_input_color_image_2_tensor_transform = T.Compose([
        T.ToTensor(),
        T.Resize(classic_input_size, interpolation=T.InterpolationMode.NEAREST)
    ])

    dataset_output_color_image_2_tensor_transform = T.Compose([
        T.ToTensor(),
        T.Resize(classic_input_size, interpolation=T.InterpolationMode.NEAREST)
    ])
elif using_Costume_Unet or using_ClassicalUnet or using_multiresunet or \
        using_fcn_50 or using_fcn_101 or using_N2N_original_unet:
    root_logger.info("Using Transform with T.Normalize")
    dataset_input_color_image_2_tensor_transform = T.Compose([
        T.ToTensor(),
        T.Resize(classic_input_size, interpolation=T.InterpolationMode.NEAREST),
        T.Normalize(mean=dataset_mean,
                    std=dataset_std),
    ])

    dataset_output_color_image_2_tensor_transform = T.Compose([
        T.ToTensor(),
        T.Resize(classic_input_size, interpolation=T.InterpolationMode.NEAREST),
        T.Normalize(mean=dataset_mean,
                    std=dataset_std),
    ])
else:
    root_logger.info("No config explicitly matched, Using Transform without T.Normalize")
    dataset_input_color_image_2_tensor_transform = T.Compose([
        T.ToTensor(),
        T.Resize(classic_input_size, interpolation=T.InterpolationMode.NEAREST)
    ])

    dataset_output_color_image_2_tensor_transform = T.Compose([
        T.ToTensor(),
        T.Resize(classic_input_size, interpolation=T.InterpolationMode.NEAREST)
    ])

transform_output_bw_image_2_tensor_transform = T.Compose([
    T.ToTensor(),
    T.Resize(classic_input_size, interpolation=T.InterpolationMode.NEAREST)
])

data_root = "data"