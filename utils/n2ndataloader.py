import csv
import glob
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class NoiseRGB_2_CleanRGB_Dataset(Dataset):
    """Offline version of v1 online version
    now this snipple is more general.
    This class return a PIL img instance,
    before training, use toTensor and Normalize transformation.
    The dataset should already be in 800x800"""

    def __init__(self, sourceDir='data/Noised_source', transform=None,
                 target_transform=None):
        self.sourceDir = f"{sourceDir}/JPEGImages"
        self.sourceList = sorted(glob.glob(os.path.join(self.sourceDir, "*.jpg"), recursive=True),
                                 key=lambda x: int(Path(x).stem))
        # log.info("Dataset created using new lambda")
        # print("Dataset Acquire source list: ", self.sourceList)
        self.targetDir = f"{sourceDir}/Originals"
        self.targetList = sorted(glob.glob(os.path.join(self.targetDir, "*.png"), recursive=True),
                                 key=lambda x: int(Path(x).stem))
        # print("Dataset Acquire target list: ", self.targetList)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sourceList)

    def __getitem__(self, idx):
        """Offline version"""
        source = (Image.open(self.sourceList[idx])).convert('RGB')
        target = (Image.open(self.targetList[idx])).convert('RGB')
        if self.transform:
            source = self.transform(source)
        if self.target_transform:
            target = self.target_transform(target)
        return source, target


class NoiseRGB_2_NoiseRGB_Dataset(Dataset):
    """Offline version of v1 online version
    now this snipple is more general.
    This class return a PIL img instance,
    before training, use toTensor and Normalize transformation.
    The dataset should already be in 800x800"""

    def __init__(self, sourceDir='data/Noised_source', targetDir='data/Noised_target', transform=None,
                 target_transform=None):
        self.sourceDir = f"{sourceDir}/JPEGImages"
        self.sourceList = sorted(glob.glob(os.path.join(self.sourceDir, "*.jpg"), recursive=True) +
                                 glob.glob(os.path.join(self.sourceDir, "*.png"), recursive=True),
                                 key=lambda x: int(Path(x).stem))
        # the list contains entities like 'data/Bodymark_Dataset/JPEGImages\\0.jpg'
        # first split base on \\ mark, then split base on . to extract integer value of file
        # I don't think this will hold for all system, linux based system might use different splitter
        # The key is used for sorting base on name
        self.targetDir = f"{targetDir}/JPEGImages"
        self.targetList = sorted(glob.glob(os.path.join(self.targetDir, "*.jpg"), recursive=True) +
                                 glob.glob(os.path.join(self.sourceDir, "*.png"), recursive=True),
                                 key=lambda x: int(Path(x).stem))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sourceList)

    def __getitem__(self, idx):
        """Offline version"""

        source = (Image.open(self.sourceList[idx])).convert('RGB')
        target = (Image.open(self.targetList[idx])).convert('RGB')
        if self.transform:
            source = self.transform(source)
        if self.target_transform:
            target = self.target_transform(target)
        return source, target


class TestDataFeeder4Classify(Dataset):
    """Prepare Data for classification
    Need label.csv in the image folder.
    Image naming convention:
    contain 'without' = no mark, label as 0
    no 'without' but has 'with' = has mark label as 1
    otherwise throw exception."""

    def __init__(self, sourceDir='data/Noised_source', transform=None):
        """There is no sub fodlers like JPEGImages
        for this dataloader.
        We need Image files, and a label CSV indexed using file basename.extension.
        ONly 2 columns is supported for now"""
        self.sourceDir = sourceDir
        self.sourceList = glob.glob(os.path.join(self.sourceDir, "*.jpg"), recursive=True)
        self.transform = transform
        self.labels_csv_reader = csv.reader(open(f"{sourceDir}/label.csv", "r", encoding='utf-8'), delimiter=",")
        self.labels_dict = {}
        """turn the label csv into dict for convenient searching"""
        for row in self.labels_csv_reader:
            image_basename, image_label = row
            self.labels_dict[image_basename] = image_label

    def __len__(self):
        return len(self.sourceList)

    def __getitem__(self, idx):
        """Offline version"""
        source = (Image.open(self.sourceList[idx])).convert('RGB')
        label = self.labels_dict[os.path.basename(self.sourceList[idx])]
        if self.transform:
            source = self.transform(source)
        return source, label


class TestDataFeeder4Classify4Training(TestDataFeeder4Classify):

    def __init__(self, sourceDir='data/Noised_source', transform=None):
        super(TestDataFeeder4Classify4Training, self).__init__(sourceDir=sourceDir, transform=transform)
        self.img_labels = pd.read_csv(f"{sourceDir}/annotations_file")

    def __getitem__(self, idx):
        """Offline version"""
        source = (Image.open(self.sourceList[idx])).convert('RGB')
        label_tensor = torch.FloatTensor([0, 0])
        label = self.labels_dict[os.path.basename(self.sourceList[idx])]
        label_tensor[int(label)] = 1
        if self.transform:
            source = self.transform(source)
        return source, label_tensor


class TestDataFeeder(NoiseRGB_2_NoiseRGB_Dataset):
    """Prepare data for u-net torch model
    This dataset serves to normal training scheme.
    When in test mode, the returned target can be omitted using _."""

    def __init__(self, sourceDir='data/raw', transform=None,
                 target_transform=None):
        super(TestDataFeeder, self).__init__(sourceDir, sourceDir, transform, target_transform)
        self.targetDir = f"{sourceDir}/SegmentationBW"
        self.targetList = sorted(glob.glob(os.path.join(self.targetDir, "*.png"), recursive=True),
                                 key=lambda x: int(Path(x).stem))
        self.cleanDir = f"{sourceDir}/Originals"
        self.cleanList = sorted(glob.glob(os.path.join(self.cleanDir, "*.png"), recursive=True),
                                key=lambda x: int(Path(x).stem))
        # print("Dataset list: ", self.sourceDir)
        # self.img_labels = pd.read_csv(f"{sourceDir}/annotations_file")

    def __getitem__(self, idx):
        """Offline version"""

        source = (Image.open(self.sourceList[idx])).convert('RGB')
        target = (Image.open(self.targetList[idx])).convert('RGB')
        clean = (Image.open(self.cleanList[idx])).convert('RGB')
        if self.transform:
            source = self.transform(source)
            clean = self.transform(clean)
        if self.target_transform:
            target = self.target_transform(target)
        return source, target, clean


class NoiseRGB_2_BW_Dataset(NoiseRGB_2_NoiseRGB_Dataset):
    """Prepare data for u-net torch model
    This dataset serves to normal training scheme.
    When in test mode, the returned target can be omitted using _."""

    def __init__(self, sourceDir='data/raw', transform=None,
                 target_transform=None):
        super(NoiseRGB_2_BW_Dataset, self).__init__(sourceDir, sourceDir, transform, target_transform)
        self.targetDir = f"{sourceDir}/SegmentationBW"
        self.targetList = sorted(glob.glob(os.path.join(self.targetDir, "*.png"), recursive=True),
                                 key=lambda x: int(Path(x).stem))
        # print("Dataset list: ", self.sourceDir)

    def __getitem__(self, idx):
        """Offline version"""
        source = (Image.open(self.sourceList[idx])).convert('RGB')
        target = (Image.open(self.targetList[idx])).convert('L')
        # conver('L') turn the mask input a one channel gray scale
        if self.transform:
            source = self.transform(source)
        if self.target_transform:
            target = self.target_transform(target)
        return source, target
