import argparse
import glob
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Run Test On model')
    parser.add_argument('--folder', '-f', dest='folder',
                        type=str, default="G:\Pytorch-QA-MG\data\Bodymark_Classify\Img with bodymark Classify",
                        help='Complete path of the JPEGImages folder inside certain dataset')
    return vars(parser.parse_args())


args = get_args()

sourceDir = Path(args['folder'])
file_names = glob.glob(os.path.join(sourceDir, "*.jpg"), recursive=True)
base_file_names = list(map(lambda x: os.path.basename(x), file_names))

train_df = pd.DataFrame(columns=["img_name", "label"])

for base_file_name in base_file_names:
    if "without" in base_file_name:
        log.info("without detected, assigning 0 to image")
        temp_dict = {'img_name': base_file_name, 'label': 0}
        train_df = train_df.append(temp_dict, ignore_index=True)
    elif "with" in base_file_name:
        log.info("with detected, assigning1 to image")
        temp_dict = {'img_name': base_file_name, 'label': 1}
        train_df = train_df.append(temp_dict, ignore_index=True)
    else:
        log.info("Nothing Detected!")
        raise Exception("Nothing Detected in filename, check if you have name them probably!")

train_df.to_csv(os.path.join(sourceDir,'label.csv'), index=False, header=True)
