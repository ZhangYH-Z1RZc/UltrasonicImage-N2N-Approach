import argparse
import json
import logging
import time


import numpy as np
from torch.utils.data import DataLoader

import config
import utils.postProcessUtils as post
import wandb
from postProces import Get_Model
from utils.n2ndataloader import TestDataFeeder4Classify




def get_args():
    parser = argparse.ArgumentParser(description='Run Classify test on model')
    parser.add_argument('--folder', '-f', dest='folder',
                        type=str, default="data/Classify_test/Bodymarker", help='Location where the csv and images are')
    parser.add_argument('--export-prefix', '-ep', dest='test_export_location_prefix',
                        type=str, default="classify_export",
                        help='where to export results')
    parser.add_argument('--checkpoint', '-c', dest='checkpoint',
                        type=str,
                        default="checkpoints/BodyMark/N2N_Costume_Unet_N2N_no_Norm_Smooth/2022-12-17_checkpoint_Bodymark_Dataset_epoch_10.pth",
                        help='path to checkpoint')
    # parser.add_argument('--enable-connectivity-filter', '-cf', dest='connectivity_filter',
    #                     type=int,
    #                     default=1,
    #                     help='use connectivity-filter or not')
    # parser.add_argument('--otsu-threshold', '-ot', dest='otsu_threshold',
    #                     type=int,
    #                     default=128,
    #                     help='the otsu_threshold for binary operation')
    # parser.add_argument('--conectivity-threshold', '-ct', dest='connectivity_threshold',
    #                     type=int,
    #                     default=10,
    #                     help='the connectivity_threshold for connectivity_filter ')
    parser.add_argument('--runname', '-rn', dest='run_name',
                        type=str,
                        default="Test_Name",
                        help='Name of Wandb RUn')
    return vars(parser.parse_args())


def Score_Computation_Loop(net, source_data_Loader, params, connectivity_filter
                           , connectivity_threshold, otsu_threshold, count_threshold=0):
    start_time = time.time()
    prediction_results = []
    # average_true_image_count = []
    # average_false_image_count = []
    for step, (source, label_of_source) in enumerate(source_data_Loader):
        step_processor = post.PostProcessor4Classify(source, net,
                                                     output_folderPrefix=params['test_export_location_prefix'],
                                                     connectivity_filter=connectivity_filter,
                                                     connectivity_threshold=connectivity_threshold,
                                                     otsu_threshold=otsu_threshold,
                                                     step=step)
        binary_result = step_processor.model_output_numpy_bool
        white_pixel_count = binary_result.sum() / 255  # the binary result is in uint8, not 0 1...need to divide it by 255
        # if int(label_of_source[0]):
        #     average_true_image_count.append(white_pixel_count)
        # else:
        #     average_false_image_count.append(white_pixel_count)
        if white_pixel_count >= count_threshold:
            """Bodymark,Vascular use >=
            Anchor use <="""
            prediction = 1
        else:
            prediction = 0
        if prediction == int(label_of_source[0]):
            prediction_results.append(1)
        else:
            prediction_results.append(0)
        # step_processor.Write_File_Of_Interest(prediction=prediction, label=int(label_of_source[0]))
        if len(prediction_results) % 10 ==0:
            # log.info(f"--- {(time.time() - start_time)} seconds --- for {len(prediction_results)} images")
            log.info(f"--- Average: {(time.time() - start_time)/len(prediction_results)} seconds/image")

    log.info(f"The accuracy is {np.array(prediction_results).sum() / len(prediction_results)}")
    # log.info(f"Average for true images {np.array(average_true_image_count).mean()}"
    #          f"Var: {np.array(average_true_image_count).var()}")
    # log.info(f"Average for false images {np.array(average_false_image_count).mean()}"
    #          f"Var: {np.array(average_false_image_count).var()}")
    log.info(f"--- {(time.time() - start_time)} seconds ---")
    return np.array(prediction_results).sum() / len(prediction_results)


def Construct_Dataset(folder: str = "classify_export"):
    """TODO
    Need to create a dataloader for those images."""
    source_data = TestDataFeeder4Classify(sourceDir=folder,
                                          transform=config.dataset_input_color_image_2_tensor_transform)  # Create DataSet
    source_data_Loader = DataLoader(source_data, batch_size=config.test_mode_batch_size, shuffle=False)
    return source_data_Loader


"""TODO
write function to compute the threshold that could get max accuracy"""


def Connectivity_looper(param, net, source_data_Loader):
    results_dict = {}
    for white_pixel_count_threshold in [10]:
        # Bodymarker 0.95: Ct14, cf 1, White COunt 30, Otsu 64
        # Measure 400
        # Vascular 750
        # log.info(f"Current Count Threshold is {white_pixel_count_threshold}")
        for otsu_threshold in [225]:
            # log.info(f"Current Otsu threshold: {otsu_threshold}")
            # 64 128 192 230
            for connectivity_filter in [1]:
                if connectivity_filter:
                    # log.info("Current Connectivity_filter is ON!")
                    for connectivity_threshold in [50]:
                        log.info("----------------------")
                        log.info(f"Current Count Threshold is {white_pixel_count_threshold}")
                        log.info(f"Current Otsu threshold: {otsu_threshold}")
                        log.info("Current Connectivity_filter is ON!")
                        log.info(f"Current Connectivity_threshold is {connectivity_threshold}")
                        acc = Score_Computation_Loop(net, source_data_Loader, param,
                                                     connectivity_filter,
                                                     connectivity_threshold,
                                                     otsu_threshold,
                                                     white_pixel_count_threshold)
                        results_dict[
                            f"white_{white_pixel_count_threshold}" +
                            f"_Otsu_{otsu_threshold}_cf_{connectivity_filter}" +
                            f"_ct_{connectivity_threshold}"] = acc
                else:
                    log.info("----------------------")
                    log.info(f"Current Count Threshold is {white_pixel_count_threshold}")
                    log.info(f"Current Otsu threshold: {otsu_threshold}")
                    log.info("connectivity_filter is Off! No Filter Threshold Loop")
                    acc = Score_Computation_Loop(net, source_data_Loader, param,
                                                 connectivity_filter,
                                                 0,
                                                 otsu_threshold,
                                                 white_pixel_count_threshold)
                    results_dict[
                        f"white_{white_pixel_count_threshold}" +
                        f"_Otsu_{otsu_threshold}_cf_{connectivity_filter}"] = acc

                with open(f'Classify_result_temp.json', 'a') as fp:
                    json.dump(results_dict, fp)
    log.info(f"Max acc is under {max(results_dict, key=results_dict.get)}"
             f"The max acc is {results_dict[max(results_dict, key=results_dict.get)]}"
             )
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open(f'Classify_result_{timestr}.json', 'w') as fp:
        json.dump(results_dict, fp)


def main(param):
    source_data_Loader = Construct_Dataset(folder=param['folder'])
    net = Get_Model(param)
    Connectivity_looper(param, net, source_data_Loader)


if __name__ == "__main__":
    args = get_args()
    wandb_experiment = wandb.init(project='Classify', resume='allow')
    wandb.run.name = args['run_name']
    log = logging.getLogger(__name__)
    main(args)
