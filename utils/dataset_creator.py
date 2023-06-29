import os

import config
from utils.n2ndataloader import NoiseRGB_2_NoiseRGB_Dataset, NoiseRGB_2_BW_Dataset, NoiseRGB_2_CleanRGB_Dataset

"""Most function in this file will be in
Object-Like Form for future multi-patch"""


class CreateDataset_ForNoiseRGBToNoiseRGB_Training:
    def __init__(self, dataPrefix):
        self.dataPrefix = dataPrefix
        self.transform_input_normal = config.dataset_input_color_image_2_tensor_transform
        self.transform_output_normal = config.dataset_output_color_image_2_tensor_transform
        self.transform_output_bw = config.transform_output_bw_image_2_tensor_transform

    def __call__(self):
        dir_source, dir_target = self.get_directory_list()
        dataset = NoiseRGB_2_NoiseRGB_Dataset(sourceDir=dir_source, targetDir=dir_target,
                                              transform=self.transform_input_normal,
                                              target_transform=self.transform_output_normal)
        return dataset

    def Check_Directory(self, directory_list):
        for directory in directory_list:
            if not os.path.isdir(directory):
                raise Exception(f"{directory} doesn't exist!")

    def get_directory_list(self):
        directory_list = [f'data/{self.dataPrefix}', f'data/{self.dataPrefix}_paired']
        dir_source = directory_list[0]
        dir_target = directory_list[1]
        return dir_source, dir_target


class CreateDataset_ForNoiseRGBToCleanRGB_Training(CreateDataset_ForNoiseRGBToNoiseRGB_Training):
    def __call__(self):
        dir_source = self.get_directory_list()
        dataset = NoiseRGB_2_CleanRGB_Dataset(sourceDir=dir_source, transform=self.transform_input_normal,
                                              target_transform=self.transform_output_normal)
        return dataset

    def get_directory_list(self):
        directory_list = [f'data/{self.dataPrefix}']
        self.Check_Directory(directory_list)
        dir_source = directory_list[0]
        return dir_source


class CreateDataset_ForNoiseRGBToBWMask_Training(CreateDataset_ForNoiseRGBToNoiseRGB_Training):
    def __call__(self):
        dir_source = self.get_directory_list()
        dataset = NoiseRGB_2_BW_Dataset(sourceDir=dir_source, transform=self.transform_input_normal,
                                        target_transform=self.transform_output_bw)
        return dataset

    def get_directory_list(self):
        directory_list = [f'data/{self.dataPrefix}']
        self.Check_Directory(directory_list)
        dir_source = directory_list[0]
        return dir_source
