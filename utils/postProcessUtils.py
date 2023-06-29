import logging
import os
from math import log10, sqrt

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from psnr_hvsm import psnr_hvs_hvsm
from scipy.ndimage import label, generate_binary_structure
from skimage.metrics import structural_similarity

import config

log = logging.getLogger(__name__)


def showNumpyImage(numpyArray):
    Image.fromarray(numpyArray).show()
    return 0


class BodyMarkPostProcessor:
    """This class assume the batch size of the dataloader to be 1
    most of the operation is performed on numpy array"""

    def __init__(self, source, truthMaskTensor, clean_img, model, output_folderPrefix: str = "Test_Export", step=0):
        self.step = step
        self.model = model
        # if config.using_gaussian_diffusion:
        #     self.diffusion_model = GaussianDiffusion(
        #         model=self.model,
        #         image_size=config.classic_input_size[0],
        #         timesteps=1000,
        #         loss_type="l1"
        #     )
        self.output_folderPrefix = output_folderPrefix

        if config.final_channel == 1:
            """Dealing with BW mask output"""
            self.model_input_tensor = source.to(device=config.device, dtype=torch.float32)  # tensor
            self.model_input_numpy_RGB = self.InNormalize_Batched_Std_Mean_Normalized_Tensor_to_RGB_Numpy(
                self.model_input_tensor)
            if config.using_fcn_50:
                self.model_output_tensor = model(self.model_input_tensor)['out']
            else:
                self.model_output_tensor = model(self.model_input_tensor)
            self.model_output_numpy_bool = self.Tensor_1Channel_BW_To_Bool_Numpy(self.model_output_tensor)
            self.truth_mask_tensor = truthMaskTensor
            self.truth_mask_numpy_bool = self.Tensor_3Channel_BW_TruthMask_To_Bool_Numpy(self.truth_mask_tensor)
        if config.final_channel == 3:
            """Dealing with SRGB output"""
            self.model_input_tensor = source.to(device=config.device, dtype=torch.float32)  # tensor
            if config.using_UNet_pp or config.using_DeepLabV3:
                log.debug("Using No STD MEAN innormalization")
                self.model_input_numpy_RGB = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_Numpy(
                    self.model_input_tensor)
            elif config.using_Costume_Unet or config.using_ClassicalUnet or config.using_multiresunet or \
                    config.using_fcn_50 or config.using_fcn_101 or config.using_N2N_original_unet:
                log.debug("Using STD MEAN innormalization")
                self.model_input_numpy_RGB = self.InNormalize_Batched_Std_Mean_Normalized_Tensor_to_RGB_Numpy(
                    self.model_input_tensor)
            else:
                log.debug("Using No STD MEAN innormalization")
                self.model_input_numpy_RGB = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_Numpy(
                    self.model_input_tensor)
            if config.using_fcn_50 or config.using_fcn_101:
                """Using FCN model from Torch HUb, they out put a dictionary,
                you need to manually assign the key word"""
                self.model_output_tensor = model(self.model_input_tensor)['out']
            elif config.using_gaussian_diffusion:
                """Dealing with Gaussian Samppling Passing input as condition to get a BW mask"""
                # self.model_output_tensor = self.diffusion_model.sample(batch_size=config.test_mode_batch_size,
                #                                                        condition_img=self.model_input_tensor)
                self.model_output_tensor = self.model.sample(batch_size=config.test_mode_batch_size,
                                                             condition_img=self.model_input_tensor)
            else:
                """Normally, the model output the result and the result only.
                for some other mode, this might be different, and it is dealt in above if-else control flow"""
                self.model_output_tensor = model(self.model_input_tensor)
            self.model_output_numpy_RGB = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_Numpy(
                self.model_output_tensor)
            self.model_output_numpy_bool = self.Binary_Result_from_Input_n_Output_Tensor_Subtraction()
            # self.model_output_numpy_bool = self.Binary_Result_from_BW_Output_Tensor()
            self.truth_mask_tensor = truthMaskTensor
            self.truth_mask_numpy_bool = self.Tensor_RGB_TruthMask_To_Bool_Numpy(self.truth_mask_tensor)
            self.clean_original_tensor = clean_img
            self.clean_original_numpy_RGB = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_Numpy(
                self.clean_original_tensor)

    def Binary_Result_from_Input_n_Output_Tensor_Subtraction(self):
        grayscaleNumpy = self.GrayScale_Numpy_from_Input_n_Output_Tensor_Subtraction()
        result = self.Otsu_Threshold_On_Grayscale_Numpy(grayscaleNumpy, thresholdValue=5)
        if config.using_conectivity_filter:
            # log.info("Using Connectivity Filter")
            result = self.Connectivty_Filter_a_Binary_Numpy(result)
        return result

    def Binary_Result_from_BW_Output_Tensor(self):
        grayScale = self.RGB_Tensor_TO_Grayscale_Numpy(self.model_output_tensor)
        result = self.Otsu_Threshold_On_Grayscale_Numpy(grayScale, thresholdValue=128)
        if config.using_conectivity_filter:
            # log.info("Using Connectivity Filter")
            result = self.Connectivty_Filter_a_Binary_Numpy(result)
        return result

    def GrayScale_Numpy_from_Input_n_Output_Tensor_Subtraction(self):
        """When subtracting 2 uint8 image, it is essential to make sure there is no overflow problem"""
        model_input_np_array = self.RGB_Tensor_TO_Grayscale_Numpy(self.model_input_tensor)
        model_output_np_array = self.RGB_Tensor_TO_Grayscale_Numpy(self.model_output_tensor)
        # showImage(model_output_np_array)
        sub_int = self.Subtract_2_NP_Array(model_input_np_array, model_output_np_array)
        return sub_int

    def Write_Numpy_toFile(self, numpyArray, indexPrefix):
        directory_path = f"{config.data_root}/{self.output_folderPrefix}"
        if not os.path.isdir(directory_path):
            log.info(f"Creating {directory_path}")
            os.makedirs(directory_path)
        cv2.imwrite(f"{directory_path}/{indexPrefix}.jpg", numpyArray)

    def Tensor_RGB_TruthMask_To_Bool_Numpy(self, truth_mask):
        truth_mask_np = self.InNormalize_Batched_Std_Mean_Normalized_Tensor_to_RGB_Numpy(truth_mask)
        truth_mask_np = self.RGB_Numpy_Img_to_Gray(truth_mask_np)
        result = self.Otsu_Threshold_On_Grayscale_Numpy(truth_mask_np)
        return result

    def Tensor_3Channel_BW_TruthMask_To_Bool_Numpy(self, truth_mask):
        unbatched_tensor = self.UnBatch_BatchedBinary_Tensor_of_BatchSize_One(truth_mask)
        aPIL = self.InNormalized_RGB_Tensor_To_RGB_PIL(unbatched_tensor)
        np_result = self.RGB_PIL_To_RGB_UINT8_Numpy(aPIL)
        truth_mask_np = self.RGB_Numpy_Img_to_Gray(np_result)
        result = self.Otsu_Threshold_On_Grayscale_Numpy(truth_mask_np, thresholdValue=128)
        return result

    def Tensor_1Channel_BW_To_Bool_Numpy(self, aTensor):
        unbatched_tensor = self.UnBatch_BatchedBinary_Tensor_of_BatchSize_One(aTensor)
        aPIL = self.GrayScale_Tensor_To_GrayScale_PIL(unbatched_tensor)
        np_result = self.GrayScale_PIL_To_GrayScale_Numpy(aPIL)
        result = self.Otsu_Threshold_On_Grayscale_Numpy(np_result, thresholdValue=240)
        return result

    def Write_File_Of_Interest(self):
        if self.Compute_Dice_Score() < 0.3 or self.Compute_Dice_Score() > 0.85:
            self.Write_Numpy_toFile(self.truth_mask_numpy_bool, f"{self.step}_TruthMask")
            self.Write_Numpy_toFile(self.model_input_numpy_RGB, f"{self.step}_modelInput_RGB")
            if config.final_channel == 3:
                self.Write_Numpy_toFile(self.model_output_numpy_bool, f"{self.step}_modelOutput_Bool")
                self.Write_Numpy_toFile(self.model_output_numpy_RGB, f"{self.step}_modelOutput_RGB")
            if config.final_channel == 1:
                self.Write_Numpy_toFile(self.model_output_numpy_bool, f"{self.step}_modelOutput_Bool")

    def Write_File_every_10_Step(self):
        if self.step % 10 == 0:
            self.Write_Numpy_toFile(self.truth_mask_numpy_bool, f"{self.step}_TruthMask")
            self.Write_Numpy_toFile(self.model_input_numpy_RGB, f"{self.step}_modelInput_RGB")
            if config.final_channel == 3:
                self.Write_Numpy_toFile(self.model_output_numpy_bool, f"{self.step}_modelOutput_Bool")
                self.Write_Numpy_toFile(self.model_output_numpy_RGB, f"{self.step}_modelOutput_RGB")
            if config.final_channel == 1:
                self.Write_Numpy_toFile(self.model_output_numpy_bool, f"{self.step}_modelOutput_Bool")

    def Compute_Score_Average_Every_X_Step(self, score_value_list, score_name: str = "Dice", step_period=500):
        if self.step % step_period == 0:
            log.info(f"Average {score_name} is {np.array(score_value_list).mean()}")
            log.info(f"Variance {score_name} is {np.array(score_value_list).var()}")
        return np.array(score_value_list).mean()

    def Compute_PSNR_HVS_M_Score(self, externalImg1=None, externalImg2=None):
        clean_img_luma = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_Luma_PIL(self.clean_original_tensor)
        if externalImg1:
            clean_img_luma = Image.fromarray(externalImg1, mode='RGB')
            clean_img_luma = clean_img_luma.convert('L')
        model_output_img_luma = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_Luma_PIL(
            self.model_output_tensor)
        if externalImg2:
            model_output_img_luma = Image.fromarray(externalImg2, mode='RGB')
            model_output_img_luma = model_output_img_luma.convert('L')
        clean_img_luma = np.array(clean_img_luma) / 255
        model_output_img_luma = np.array(model_output_img_luma) / 255
        psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(clean_img_luma, model_output_img_luma)
        return psnr_hvsm

    def Compute_PSNR_Score(self, externalImg1=None, externalImg2=None):
        im1 = np.asarray(self.clean_original_numpy_RGB)  # original
        if externalImg1:
            im1 = np.asarray(externalImg1).astype(np.bool)
        im2 = np.asarray(self.model_output_numpy_RGB)  # model output
        if externalImg2:
            im2 = np.asarray(externalImg2).astype(np.bool)

        mse = np.mean((im1 - im2) ** 2)
        if mse == 0:  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def Compute_SSIM_Score(self, externalImg1=None, externalImg2=None):
        im1 = np.asarray(self.clean_original_numpy_RGB)  # original
        if externalImg1:
            im1 = np.asarray(externalImg1).astype(np.bool)
        im2 = np.asarray(self.model_output_numpy_RGB)  # model output
        if externalImg2:
            im2 = np.asarray(externalImg2).astype(np.bool)

        SSIM = structural_similarity(im1, im2, channel_axis=2)  # multichannel=True,
        return SSIM

    def Compute_Dice_Score(self, externalImg1=None, externalImg2=None, empty_score=1.0):
        """If no external img provide, then use class image for computation"""
        im1 = np.asarray(self.model_output_numpy_bool).astype(np.bool)
        if externalImg1:
            im1 = np.asarray(externalImg1).astype(np.bool)
        im2 = np.asarray(self.truth_mask_numpy_bool).astype(np.bool)
        if externalImg2:
            im2 = np.asarray(externalImg2).astype(np.bool)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return empty_score

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        return 2. * intersection.sum() / im_sum

    def Compute_IoU_Score(self, externalImg1=None, externalImg2=None, empty_score=1.0):
        im1 = np.asarray(self.model_output_numpy_bool).astype(np.bool)
        if externalImg1:
            im1 = np.asarray(externalImg1).astype(np.bool)
        im2 = np.asarray(self.truth_mask_numpy_bool).astype(np.bool)
        if externalImg2:
            im2 = np.asarray(externalImg2).astype(np.bool)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return empty_score

        # Compute IoU coefficient
        intersection = np.logical_and(im1, im2)
        union = np.logical_or(im1, im2)
        return intersection.sum() / union.sum()

    def Compute_PA_Score(self, externalImg1=None, externalImg2=None, empty_score=1.0):
        im1 = np.asarray(self.model_output_numpy_bool).astype(np.bool)
        if externalImg1:
            im1 = np.asarray(externalImg1).astype(np.bool)
        a_truth_mask = np.asarray(self.truth_mask_numpy_bool).astype(np.bool)
        if externalImg2:
            a_truth_mask = np.asarray(externalImg2).astype(np.bool)
        if im1.shape != a_truth_mask.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + a_truth_mask.sum()
        if im_sum == 0:
            return empty_score

        # Compute PA coefficient
        # intersection = np.logical_and(im1, a_truth_mask)
        positive_correct_classified = np.logical_and(im1, a_truth_mask)

        inversed_im1 = 1 - im1
        inversed_a_tturh_mask = 1 - a_truth_mask
        negative_correct_classified = np.logical_and(inversed_im1, inversed_a_tturh_mask)

        return (positive_correct_classified.sum() + negative_correct_classified.sum()) / np.ones(shape=im1.shape).sum()

    def GrayScale_PIL_To_GrayScale_Numpy(self, aPILImage):
        assert aPILImage.mode == 'L', "aPILImage is not grayscale"
        result = np.array(aPILImage, dtype=np.uint8)
        return result

    def GrayScale_Tensor_To_GrayScale_PIL(self, aTensor):
        assert aTensor.shape[0] == 1, (
                "aTensor dose not have channels of GrayScale " +
                "but InNormalized_GrayScale_Tensor_To_GrayScale_PIL is intended to deal Single Channel color space")
        transform = transforms.ToPILImage(mode="L")
        result = transform(aTensor)
        return result

    def UnBatch_BatchedBinary_Tensor_of_BatchSize_One(self, aBatchOFTensor):
        assert len(aBatchOFTensor.shape) == 4, "aBatchOFTensor is not in B,C,H,W format, dimension mismatching"
        assert aBatchOFTensor.shape[0] == 1, "aBatchOFTensor dose not have batch size of 1"
        return aBatchOFTensor.squeeze(dim=0)

    def Inverse_Normalization_for_RGB_Tensor(self, aRGBTensor):
        transform = transforms.Normalize(mean=config.dataset_inverse_mean,
                                         std=config.dataset_inverse_std)
        assert aRGBTensor.shape[0] == 3, "The input tensor dose not have RGB channels"
        result = transform(aRGBTensor)
        if config.using_gaussian_diffusion:
            log.info("Using Gaussian diffusion, no STD MEAN normalization, return original tensor")
            result = aRGBTensor
        return result

    def InNormalize_Batched_Std_Mean_Normalized_Tensor_to_RGB_Numpy(self, aTensor):
        # if config.using_fcn_50:
        #     aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        # else:
        #     aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        aTensor = self.Inverse_Normalization_for_RGB_Tensor(aTensor)
        aPIL = self.InNormalized_RGB_Tensor_To_RGB_PIL(aTensor)
        result = self.RGB_PIL_To_RGB_UINT8_Numpy(aPIL)
        return result

    def RGB_PIL_To_RGB_UINT8_Numpy(self, aPILImage):
        assert aPILImage.mode == 'RGB', "aPILImage is not a RGB Image"
        result = np.array(aPILImage, dtype=np.uint8)
        return result

    def InNormalized_RGB_Tensor_To_RGB_PIL(self, aTensor):
        assert aTensor.shape[0] == 3, (
                "aTensor dose not have channels of RGB " +
                "but InNormalized_RGB_Tensor_To_RGB_PIL is intended to deal RGB color space")
        transform = transforms.ToPILImage()
        result = transform(aTensor)
        return result

    def UnBatch_BCHW_Tensor_of_BatchSize_One(self, aBatchOFTensor):
        assert len(aBatchOFTensor.shape) == 4, "aBatchOFTensor is not in B,C,H,W format, dimension mismatching"
        assert aBatchOFTensor.shape[0] == 1, "aBatchOFTensor dose not have batch size of 1"
        return aBatchOFTensor.squeeze(dim=0)

    def InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_PIL(self, aTensor):
        aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        result = self.InNormalized_RGB_Tensor_To_RGB_PIL(aTensor)
        return result

    def RGB_PIL_Img_to_Luma(self, aRgbPIL):
        return aRgbPIL.convert('L')

    def InNormalize_Batched_ToTensor_Normalized_Tensor_to_Luma_PIL(self, aTensor):
        aRgbPIL = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_PIL(aTensor)
        LumaPil = self.RGB_PIL_Img_to_Luma(aRgbPIL)
        return LumaPil

    def RGB_Numpy_Img_to_Gray(self, numpyArray):
        assert numpyArray.dtype == np.uint8, (
            "numpyArray's data type is not np.uint8"
        )
        assert numpyArray.shape[2] == 3, (
            "numpyArray's dose not have 3 channel for RGB representation"
        )
        result = cv2.cvtColor(numpyArray, cv2.COLOR_BGR2GRAY)
        return result

    def Otsu_Threshold_On_Grayscale_Numpy(self, numpyArray, thresholdValue=10):
        assert numpyArray.dtype == np.uint8, (
            "numpyArray's data type is not np.uint8"
        )
        result = cv2.threshold(numpyArray, thresholdValue, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return result

    def Connectivty_Filter_a_Binary_Numpy(self, numpyArray, threshold=5):
        s = generate_binary_structure(2, 2)
        labeled_array, num_features = label(numpyArray, structure=s)
        for current_label in range(num_features):
            current_label = current_label + 1
            featureSum = labeled_array[np.where(labeled_array == current_label)].sum() / current_label
            if featureSum <= threshold:
                numpyArray[np.where(labeled_array == current_label)] = 0
        return numpyArray

    def RGB_Tensor_TO_Grayscale_Numpy(self, aSourceTENSOR):
        inputTensor = self.InNormalize_Batched_Std_Mean_Normalized_Tensor_to_RGB_Numpy(aSourceTENSOR)
        result = self.RGB_Numpy_Img_to_Gray(inputTensor)
        return result

    def Subtract_2_NP_Array(self, an_Input_Numpy_Array, secondary_Numpy_Array):
        sub_int = np.subtract(an_Input_Numpy_Array.astype(int), secondary_Numpy_Array.astype(int))
        sub_int[np.where(sub_int < 0)] = 0
        sub_int[np.where(sub_int > 255)] = 255
        sub_int = sub_int.astype(np.uint8)
        return sub_int

    def InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_Numpy(self, aTensor):
        # if config.using_fcn_50:
        #     aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        # else:
        #     aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        aPIL = self.InNormalized_RGB_Tensor_To_RGB_PIL(aTensor)
        result = self.RGB_PIL_To_RGB_UINT8_Numpy(aPIL)
        return result


class PostProcessor4Classify:
    def __init__(self, source, model, output_folderPrefix: str = "Test_Export",
                 connectivity_filter=1, connectivity_threshold=10, otsu_threshold=128, step=0):
        self.step = step
        self.model = model
        self.connectivity_filter = connectivity_filter
        self.connectivity_threshold = connectivity_threshold
        self.otsu_threshold = otsu_threshold
        self.output_folderPrefix = output_folderPrefix
        self.model_input_tensor = source.to(device=config.device, dtype=torch.float32)  # tensor
        log.debug("Using No STD MEAN innormalization")
        self.model_input_numpy_RGB = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_Numpy(
            self.model_input_tensor)
        """Normally, the model output the result and the result only.
        for some other mode, this might be different, and it is dealt in above if-else control flow"""
        self.model_output_tensor = model(self.model_input_tensor)
        self.model_output_numpy_RGB = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_Numpy(
            self.model_output_tensor)
        self.model_output_numpy_bool = self.Binary_Result_from_Input_n_Output_Tensor_Subtraction()

    def Binary_Result_from_Input_n_Output_Tensor_Subtraction(self):
        grayscaleNumpy = self.GrayScale_Numpy_from_Input_n_Output_Tensor_Subtraction()
        result = self.Otsu_Threshold_On_Grayscale_Numpy(grayscaleNumpy, thresholdValue=self.otsu_threshold)
        # old thresholdValue = 5
        if self.connectivity_filter:
            # log.info("Using Connectivity Filter")
            result = self.Connectivty_Filter_a_Binary_Numpy(result, threshold=self.connectivity_threshold)
            # old value: 5 - 12
        return result

    def GrayScale_Numpy_from_Input_n_Output_Tensor_Subtraction(self):
        """When subtracting 2 uint8 image, it is essential to make sure there is no overflow problem"""
        model_input_np_array = self.RGB_Tensor_TO_Grayscale_Numpy(self.model_input_tensor)
        model_output_np_array = self.RGB_Tensor_TO_Grayscale_Numpy(self.model_output_tensor)
        # showImage(model_output_np_array)
        sub_int = self.Subtract_2_NP_Array(model_input_np_array, model_output_np_array)
        return sub_int

    def Write_Numpy_toFile(self, numpyArray, indexPrefix, color_mode):
        directory_path = f"{config.data_root}/{self.output_folderPrefix}"
        if not os.path.isdir(directory_path):
            log.info(f"Creating {directory_path}")
            os.makedirs(directory_path)
        if color_mode == "rgb":
            cv2.imwrite(f"{directory_path}/{indexPrefix}.jpg", cv2.cvtColor(numpyArray, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(f"{directory_path}/{indexPrefix}.jpg", numpyArray)

    def Tensor_RGB_TruthMask_To_Bool_Numpy(self, truth_mask):
        truth_mask_np = self.InNormalize_Batched_Std_Mean_Normalized_Tensor_to_RGB_Numpy(truth_mask)
        truth_mask_np = self.RGB_Numpy_Img_to_Gray(truth_mask_np)
        result = self.Otsu_Threshold_On_Grayscale_Numpy(truth_mask_np)
        return result

    def Tensor_3Channel_BW_TruthMask_To_Bool_Numpy(self, truth_mask):
        unbatched_tensor = self.UnBatch_BatchedBinary_Tensor_of_BatchSize_One(truth_mask)
        aPIL = self.InNormalized_RGB_Tensor_To_RGB_PIL(unbatched_tensor)
        np_result = self.RGB_PIL_To_RGB_UINT8_Numpy(aPIL)
        truth_mask_np = self.RGB_Numpy_Img_to_Gray(np_result)
        result = self.Otsu_Threshold_On_Grayscale_Numpy(truth_mask_np, thresholdValue=128)
        return result

    def Tensor_1Channel_BW_To_Bool_Numpy(self, aTensor):
        unbatched_tensor = self.UnBatch_BatchedBinary_Tensor_of_BatchSize_One(aTensor)
        aPIL = self.GrayScale_Tensor_To_GrayScale_PIL(unbatched_tensor)
        np_result = self.GrayScale_PIL_To_GrayScale_Numpy(aPIL)
        result = self.Otsu_Threshold_On_Grayscale_Numpy(np_result, thresholdValue=240)
        return result

    def Write_File_Of_Interest(self, prediction, label):
        # self.Write_Numpy_toFile(self.truth_mask_numpy_bool, f"{self.step}_TruthMask")
        self.Write_Numpy_toFile(self.model_input_numpy_RGB, f"{self.step}_modelInput_RGB_p_{prediction}_l_{label}",
                                "rgb")
        self.Write_Numpy_toFile(self.model_output_numpy_bool, f"{self.step}_modelOutput_Bool_p_{prediction}_l_"
                                                              f"{label}","bool")
        self.Write_Numpy_toFile(self.model_output_numpy_RGB, f"{self.step}_modelOutput_RGB_p_{prediction}_l_{label}",
                                "rgb")

    def Write_File_every_10_Step(self):
        if self.step % 10 == 0:
            # self.Write_Numpy_toFile(self.truth_mask_numpy_bool, f"{self.step}_TruthMask")
            self.Write_Numpy_toFile(self.model_input_numpy_RGB, f"{self.step}_modelInput_RGB")
            if config.final_channel == 3:
                self.Write_Numpy_toFile(self.model_output_numpy_bool, f"{self.step}_modelOutput_Bool")
                self.Write_Numpy_toFile(self.model_output_numpy_RGB, f"{self.step}_modelOutput_RGB")
            if config.final_channel == 1:
                self.Write_Numpy_toFile(self.model_output_numpy_bool, f"{self.step}_modelOutput_Bool")

    def GrayScale_PIL_To_GrayScale_Numpy(self, aPILImage):
        assert aPILImage.mode == 'L', "aPILImage is not grayscale"
        result = np.array(aPILImage, dtype=np.uint8)
        return result

    def GrayScale_Tensor_To_GrayScale_PIL(self, aTensor):
        assert aTensor.shape[0] == 1, (
                "aTensor dose not have channels of GrayScale " +
                "but InNormalized_GrayScale_Tensor_To_GrayScale_PIL is intended to deal Single Channel color space")
        transform = transforms.ToPILImage(mode="L")
        result = transform(aTensor)
        return result

    def UnBatch_BatchedBinary_Tensor_of_BatchSize_One(self, aBatchOFTensor):
        assert len(aBatchOFTensor.shape) == 4, "aBatchOFTensor is not in B,C,H,W format, dimension mismatching"
        assert aBatchOFTensor.shape[0] == 1, "aBatchOFTensor dose not have batch size of 1"
        return aBatchOFTensor.squeeze(dim=0)

    def Inverse_Normalization_for_RGB_Tensor(self, aRGBTensor):
        transform = transforms.Normalize(mean=config.dataset_inverse_mean,
                                         std=config.dataset_inverse_std)
        assert aRGBTensor.shape[0] == 3, "The input tensor dose not have RGB channels"
        result = transform(aRGBTensor)
        if config.using_gaussian_diffusion:
            log.info("Using Gaussian diffusion, no STD MEAN normalization, return original tensor")
            result = aRGBTensor
        return result

    def InNormalize_Batched_Std_Mean_Normalized_Tensor_to_RGB_Numpy(self, aTensor):
        # if config.using_fcn_50:
        #     aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        # else:
        #     aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        aTensor = self.Inverse_Normalization_for_RGB_Tensor(aTensor)
        aPIL = self.InNormalized_RGB_Tensor_To_RGB_PIL(aTensor)
        result = self.RGB_PIL_To_RGB_UINT8_Numpy(aPIL)
        return result

    def RGB_PIL_To_RGB_UINT8_Numpy(self, aPILImage):
        assert aPILImage.mode == 'RGB', "aPILImage is not a RGB Image"
        result = np.array(aPILImage, dtype=np.uint8)
        return result

    def InNormalized_RGB_Tensor_To_RGB_PIL(self, aTensor):
        assert aTensor.shape[0] == 3, (
                "aTensor dose not have channels of RGB " +
                "but InNormalized_RGB_Tensor_To_RGB_PIL is intended to deal RGB color space")
        transform = transforms.ToPILImage()
        result = transform(aTensor)
        return result

    def UnBatch_BCHW_Tensor_of_BatchSize_One(self, aBatchOFTensor):
        assert len(aBatchOFTensor.shape) == 4, "aBatchOFTensor is not in B,C,H,W format, dimension mismatching"
        assert aBatchOFTensor.shape[0] == 1, "aBatchOFTensor dose not have batch size of 1"
        return aBatchOFTensor.squeeze(dim=0)

    def InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_PIL(self, aTensor):
        aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        result = self.InNormalized_RGB_Tensor_To_RGB_PIL(aTensor)
        return result

    def RGB_PIL_Img_to_Luma(self, aRgbPIL):
        return aRgbPIL.convert('L')

    def InNormalize_Batched_ToTensor_Normalized_Tensor_to_Luma_PIL(self, aTensor):
        aRgbPIL = self.InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_PIL(aTensor)
        LumaPil = self.RGB_PIL_Img_to_Luma(aRgbPIL)
        return LumaPil

    def RGB_Numpy_Img_to_Gray(self, numpyArray):
        assert numpyArray.dtype == np.uint8, (
            "numpyArray's data type is not np.uint8"
        )
        assert numpyArray.shape[2] == 3, (
            "numpyArray's dose not have 3 channel for RGB representation"
        )
        result = cv2.cvtColor(numpyArray, cv2.COLOR_BGR2GRAY)
        return result

    def Otsu_Threshold_On_Grayscale_Numpy(self, numpyArray, thresholdValue=10):
        assert numpyArray.dtype == np.uint8, (
            "numpyArray's data type is not np.uint8"
        )
        result = cv2.threshold(numpyArray, thresholdValue, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return result

    def Connectivty_Filter_a_Binary_Numpy(self, numpyArray, threshold=8):
        # old threshold = 5 then 12
        s = generate_binary_structure(2, 2)
        labeled_array, num_features = label(numpyArray, structure=s)
        for current_label in range(num_features):
            current_label = current_label + 1
            featureSum = labeled_array[np.where(labeled_array == current_label)].sum() / current_label
            if featureSum <= threshold:
                numpyArray[np.where(labeled_array == current_label)] = 0
        return numpyArray

    def RGB_Tensor_TO_Grayscale_Numpy(self, aSourceTENSOR):
        inputTensor = self.InNormalize_Batched_Std_Mean_Normalized_Tensor_to_RGB_Numpy(aSourceTENSOR)
        result = self.RGB_Numpy_Img_to_Gray(inputTensor)
        return result

    def Subtract_2_NP_Array(self, an_Input_Numpy_Array, secondary_Numpy_Array):
        sub_int = np.subtract(an_Input_Numpy_Array.astype(int), secondary_Numpy_Array.astype(int))
        sub_int[np.where(sub_int < 0)] = 0
        sub_int[np.where(sub_int > 255)] = 255
        sub_int = sub_int.astype(np.uint8)
        return sub_int

    def InNormalize_Batched_ToTensor_Normalized_Tensor_to_RGB_Numpy(self, aTensor):
        # if config.using_fcn_50:
        #     aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        # else:
        #     aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        aTensor = self.UnBatch_BCHW_Tensor_of_BatchSize_One(aTensor)
        aPIL = self.InNormalized_RGB_Tensor_To_RGB_PIL(aTensor)
        result = self.RGB_PIL_To_RGB_UINT8_Numpy(aPIL)
        return result
