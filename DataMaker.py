import argparse
import glob
import itertools
import logging
import multiprocessing
import os
import random
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

import config
from dashedlines import DashedImageDraw

NUM_OF_CPU = os.cpu_count()
log = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Customize generated images')
    parser.add_argument('--data-root', '-d', dest='data_root',
                        type=str, default="data", help='Root location of ')
    parser.add_argument('--prefix', '-p', dest='prefix',
                        type=str, default="Test_gen_data",
                        help='Prefix of the subfolder name')
    parser.add_argument('--raw-path', '-r', dest='raw_path',
                        type=str, default="Img Without Bodaymark - Combined HNR results and Provided",
                        help='Root path of clean image')
    parser.add_argument('--noise-path', '-rn', dest='noise_path',
                        type=str, default="Noise Img - Bodymark",
                        help='Root path of noise entities')
    parser.add_argument('--ratio', '-ra', dest='ratio',
                        type=int, default=50, help='how many noisy output per image')
    parser.add_argument('--name', '-na', dest='obj_name',
                        type=str, default="body_mark", help='name of object,body_mark, color_rect,measure_box')
    parser.add_argument('--num-object', '-no', dest='num_object',
                        type=int, default=1, help='number of object')
    parser.add_argument('--is-segmented', '-sg', dest='is_seged',
                        type=int, default=1, help="generating seg ground truth or not")
    parser.add_argument('--n2nmode', '-n2n', dest='n2n_enabled',
                        type=int, default=1, help="generating under n2n paired mode")
    return parser.parse_args()


class ObjectLocationInfo:
    def __init__(self, bndbox: list, objName: str):
        self.xmin = bndbox[0]
        self.xmax = bndbox[1]
        self.ymin = bndbox[2]
        self.ymax = bndbox[3]
        self.name = objName


class FolderPaths:
    def __init__(self, dataroot, prefix):
        self.JPEGImages = f"{dataroot}/{prefix}/JPEGImages"
        self.SegmentationBW = f"{dataroot}/{prefix}/SegmentationBW"
        self.SegmentationClass = f"{dataroot}/{prefix}/SegmentationClass"
        self.Annotations = f"{dataroot}/{prefix}/Annotations"
        self.CleanOrigin = f"{dataroot}/{prefix}/Originals"
        log.info(
            f"{prefix} Folders: {[self.JPEGImages, self.SegmentationBW, self.SegmentationClass, self.Annotations, self.CleanOrigin]}")
        for pathX in [self.JPEGImages, self.SegmentationBW, self.SegmentationClass, self.Annotations, self.CleanOrigin]:
            if not os.path.isdir(pathX):
                log.info(f"Creating {pathX}")
                os.makedirs(pathX)


class BodyMarkNoise:
    def __init__(self, noiseImg, targetImg, xRange=(0.375, 0.625), yRange=(0.375, 0.625)):
        self.standardSize = config.costume_input_size
        self.noiseSize = noiseImg.size

        self.noiseImg = noiseImg  # foreground noise
        self.noiseImg = self.noiseImg.convert('RGBA')
        """Make sure it is RGBA"""

        self.targetImg = targetImg.convert('RGBA')  # background
        self.resultImg = self.targetImg.copy()

        self.noisePlaceRange = (int(xRange[0] * self.standardSize[0]),
                                int(xRange[1] * self.standardSize[0]),
                                int(yRange[0] * self.standardSize[1]),
                                int(yRange[1] * self.standardSize[1]))  # xmin xmax ymin ymax
        # area that should not have noise on them
        self.TargetImage_Resize_Ratio = (1.0, 1.0)
        self.bondbox = [0, 0, 0, 0]

        self.groundTruth_Black_Background = Image.new("RGB", self.standardSize, (0, 0, 0))
        self.groundTruthBW = self.groundTruth_Black_Background.copy()
        self.groundTruthColor = self.groundTruth_Black_Background.copy()
        self.colorfulGroundTruthColor = (128, 0, 0)

    def ResizeTargetImage_2_StandardSize(self):
        width, height = self.targetImg.size
        # logger.info(f"original img size: {width}, {height}")
        self.targetImg = self.targetImg.resize(self.standardSize)
        self.TargetImage_Resize_Ratio = (self.standardSize[0] / width, self.standardSize[1] / height)
        # logger.info(f"resize scale factor: {self.originalResizeFactor}")
        self.resultImg = self.targetImg.copy()  # change of target img should be relected to resultImg

    def ResizeNoiseImage_According2_TargetImageRatio(self):
        self.noiseSize = (int(self.noiseSize[0] * self.TargetImage_Resize_Ratio[0]),
                          int(self.noiseSize[1] * self.TargetImage_Resize_Ratio[1]))
        self.noiseImg = self.noiseImg.resize(self.noiseSize)

    def Append_1_Noise(self):
        log.info("Parent Append_1_Noise Called")
        available_x_ranges = list(itertools.chain(range(50, self.noisePlaceRange[0]),
                                                  range(self.noisePlaceRange[1], self.standardSize[0])))
        xLocation = random.choice(available_x_ranges)
        available_y_ranges = list(itertools.chain(range(50, self.noisePlaceRange[2]),
                                                  range(self.noisePlaceRange[3], self.standardSize[1])))
        yLocation = random.choice(available_y_ranges)
        # xLocation = random.randrange(self.noisePlaceRange[0], self.noisePlaceRange[1])
        # yLocation = random.randrange(self.noisePlaceRange[2], self.noisePlaceRange[3])
        self.bondbox[0] = xLocation
        self.bondbox[1] = xLocation + self.noiseSize[0]
        self.bondbox[2] = yLocation
        self.bondbox[3] = yLocation + self.noiseSize[1]
        for index in range(0, 4):
            """make sure bond box doesn't reach out of the canvas"""
            if self.bondbox[index] > self.standardSize[index % 2]:
                self.bondbox[index] = self.standardSize[index % 2]
        self.resultImg.paste(self.noiseImg, (xLocation, yLocation), self.noiseImg)
        # don't re-assign after paste...

    def AppendNoise(self):
        self.Append_1_Noise()

    def CreateBWGroundTruth(self):
        whiteForeground = Image.new("RGB", self.noiseSize, (255, 255, 255))
        self.groundTruthBW.paste(whiteForeground, (self.bondbox[0], self.bondbox[2]), self.noiseImg)

    def CreateColorfulGroundTruth(self):
        colorForeground = Image.new("RGB", self.noiseSize, self.colorfulGroundTruthColor)
        self.groundTruthColor.paste(colorForeground, (self.bondbox[0], self.bondbox[2]), self.noiseImg)


class RectangleNoise(BodyMarkNoise):
    def __init__(self, noiseImg, targetImg, xRange=(0.375, 0.625), yRange=(0.375, 0.625)):
        super().__init__(noiseImg, targetImg, xRange, yRange)
        self.colorfulGroundTruthColor = (0, 128, 0)

    def noiseSize_add_randomness(self, xFactor: float = 0.5, yFactor: float = 0.5):
        widthFactor = random.uniform(1.0 - xFactor, 1.0 + xFactor)
        heightFactor = random.uniform(1.0 - yFactor, 1.0 + yFactor)
        noiseRatioTweakedSize = (int(1 + self.noiseSize[0] * widthFactor),
                                 int(1 + self.noiseSize[1] * heightFactor))
        self.noiseImg = self.noiseImg.resize(noiseRatioTweakedSize)
        self.noiseSize = self.noiseImg.size

    def AppendNoise(self):
        self.noiseSize_add_randomness()
        self.Append_1_Noise()


class MeasureBoxNoise(BodyMarkNoise):
    def __init__(self, noiseImg, targetImg, xRange=(0.375, 0.625), yRange=(0.375, 0.625)):
        super().__init__(noiseImg, targetImg, xRange, yRange)
        self.colorfulGroundTruthColor = (0, 0, 128)


class MeasureAnchorNoise(BodyMarkNoise):
    def __init__(self, noiseImg, targetImg, noise_list, xRange=(0.4, 0.6), yRange=(0.4, 0.6)):
        super().__init__(noiseImg, targetImg, xRange, yRange)
        self.noise_list = noise_list
        self.colorfulGroundTruthColor = (128, 0, 128)
        self.random_dash_length_config = (random.randint(1, 8), random.randint(1, 8))
        self.random_dash_color_config = (
            random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def AppendNoise(self):
        # Amount of anchor in controled in main body script
        for _ in range(0, 4):
            log.info("Anchor.AppendNoise called, this should be called, something is wrong")
            self.Append_1_Noise()
            self.Append_1_Noise()
            self.Append_1_Noise()
            self.Append_1_Noise()

    def Draw_Dashed_Lines(self, xy_indexes):
        dashed_drawer = DashedImageDraw(self.resultImg)

        dashed_drawer.dashed_line(xy_indexes, width=1, dash=self.random_dash_length_config,
                                  fill=self.random_dash_color_config)

    def Creat_Dashed_Line_On_TuthMask(self, xy_indexes):
        BW_dashed_drawer = DashedImageDraw(self.groundTruthBW)
        BW_dashed_drawer.dashed_line(xy_indexes, width=1, dash=self.random_dash_length_config,
                                     fill=(255, 255, 255))
        Color_dashed_drawer = DashedImageDraw(self.groundTruthColor)
        Color_dashed_drawer.dashed_line(xy_indexes, width=1, dash=self.random_dash_length_config,
                                        fill=self.colorfulGroundTruthColor)

    def Append_1_Noise(self):
        available_x_ranges = list(range(self.noisePlaceRange[0], self.noisePlaceRange[1]))
        xLocation = random.choice(available_x_ranges)
        available_y_ranges = list(range(self.noisePlaceRange[2], self.noisePlaceRange[3]))
        yLocation = random.choice(available_y_ranges)
        self.bondbox[0] = xLocation
        self.bondbox[1] = xLocation + self.noiseSize[0]
        self.bondbox[2] = yLocation
        self.bondbox[3] = yLocation + self.noiseSize[1]
        for index in range(0, 4):
            """make sure bond box doesn't reach out of the canvas"""
            if self.bondbox[index] > self.standardSize[index % 2]:
                self.bondbox[index] = self.standardSize[index % 2]
        if random.choice([1,0]):
            new_noise = Image.open(random.choice(self.noise_list))
            new_noise = new_noise.convert('RGBA')
            self.resultImg.paste(new_noise, (xLocation, yLocation), new_noise)
        else:
            self.resultImg.paste(self.noiseImg, (xLocation, yLocation), self.noiseImg)


class VascularFlowSampleBox(BodyMarkNoise):
    def __init__(self, targetImg, xRange=(0, 0.1), yRange=(0, 0.1)):
        self.standardSize = config.costume_input_size
        self.colorfulGroundTruthColor = (0, 128, 128)
        self.random_color_config = (
            random.randint(50, 255), random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.random_color_config = random.choice([(183, 188, 148,255),
                                                  (182, 206, 184,255),
                                                  (91, 115, 89,255),
                                                  (22, 60, 63,255),
                                                  ])

        self.polygon_vertices = (
            (random.randint(int(self.standardSize[0] * 0.2), int(self.standardSize[0] * 0.3)),
             random.randint(int(self.standardSize[0] * 0.2), int(self.standardSize[0] * 0.3))),  # top left
            (random.randint(int(self.standardSize[1] * 0.7), int(self.standardSize[1] * 0.8)),
             random.randint(int(self.standardSize[1] * 0.7), int(self.standardSize[1] * 0.8)))  # bottom right
        )
        noiseImg = Image.new('RGBA', config.costume_input_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(noiseImg)
        # draw.polygon(xy=self.polygon_vertices, fill=None, outline=self.random_color_config)
        draw.rectangle(xy=self.polygon_vertices, fill=None, outline=self.random_color_config)
        super().__init__(noiseImg, targetImg, xRange, yRange)

    def ResizeNoiseImage_According2_TargetImageRatio(self):
        """DO Nothing No need to resize"""
        self.noiseImg = self.noiseImg

    def Append_1_Noise(self):
        xLocation = 0
        yLocation = 0

        self.bondbox[0] = xLocation
        self.bondbox[1] = xLocation + self.noiseSize[0]
        self.bondbox[2] = yLocation
        self.bondbox[3] = yLocation + self.noiseSize[1]
        for index in range(0, 4):
            """make sure bond box doesn't reach out of the canvas"""
            if self.bondbox[index] > self.standardSize[index % 2]:
                self.bondbox[index] = self.standardSize[index % 2]
        self.resultImg.paste(self.noiseImg, (xLocation, yLocation), self.noiseImg)


def ImageMaker_ImageDraw(imgPath1, img2PathList, objectName , NoiseSourceList):
    annoteList = []
    noise_position = []
    im1 = Image.open(imgPath1)
    for imgPath2 in img2PathList:
        im2 = Image.open(imgPath2)
        im2 = im2.convert('RGBA')

        if objectName == "body_mark":
            imgGen = BodyMarkNoise(im2, im1)
        elif objectName == "color_rect":
            imgGen = RectangleNoise(im2, im1)
        elif objectName == "measure_box":
            imgGen = MeasureBoxNoise(im2, im1)
        elif objectName == "measure_anchor":
            imgGen = MeasureAnchorNoise(im2, im1, noise_list = NoiseSourceList)
        elif objectName == "vascular_flow":
            """vascular_flow do not need external noises
            just open some other noise folder with it"""
            imgGen = VascularFlowSampleBox(im1)
        else:
            raise Exception("Object class not implemented. imgGen not created.")

        imgGen.ResizeTargetImage_2_StandardSize()
        imgGen.ResizeNoiseImage_According2_TargetImageRatio()
        if objectName == "measure_anchor":
            for _ in range(0, random.randint(1,6)):  # add 4 anchor point
                imgGen.Append_1_Noise()
                annotation = ObjectLocationInfo(bndbox=imgGen.bondbox, objName=objectName)
                annoteList.append(annotation)
                pair_of_position = (
                    (imgGen.bondbox[0] + imgGen.bondbox[1]) / 2,
                    (imgGen.bondbox[2] + imgGen.bondbox[3]) / 2)

                noise_position.append(pair_of_position)
                imgGen.CreateBWGroundTruth()
                imgGen.CreateColorfulGroundTruth()
            if random.choice([1,0]):
                imgGen.Draw_Dashed_Lines(noise_position)
                imgGen.Creat_Dashed_Line_On_TuthMask(noise_position)
        else:
            imgGen.AppendNoise()
            annotation = ObjectLocationInfo(bndbox=imgGen.bondbox, objName=objectName)
            annoteList.append(annotation)
            imgGen.CreateBWGroundTruth()
            imgGen.CreateColorfulGroundTruth()

        """Make Sure the colorspace is RGB"""
        imgGen.resultImg = imgGen.resultImg.convert('RGB')
        imgGen.groundTruthBW = imgGen.groundTruthBW.convert('RGB')
        imgGen.groundTruthColor = imgGen.groundTruthColor.convert('RGB')
    return im1, imgGen.resultImg, imgGen.groundTruthBW, imgGen.groundTruthColor, annoteList


def CreateXMLData(picLocation: str, picFilename: str, picObjects: list, picWidth: int = 800, picHeight: int = 800,
                  picDepth: int = 3,
                  picSeged: int = 0):
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = picLocation
    filename = ET.SubElement(annotation, 'filename')
    filename.text = picFilename
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'GN DATABASE'
    source_annotation = ET.SubElement(source, 'annotation')
    source_annotation.text = 'GN 2022'
    image = ET.SubElement(source, 'image')
    image.text = 'GN'
    owner = ET.SubElement(annotation, 'owner')
    owner_name = ET.SubElement(owner, 'name')
    owner_name.text = 'GN'
    size = ET.SubElement(annotation, 'size')
    size_width = ET.SubElement(size, 'width')
    size_height = ET.SubElement(size, 'height')
    size_depth = ET.SubElement(size, 'depth')
    size_width.text = str(picWidth)
    size_height.text = str(picHeight)
    size_depth.text = str(picDepth)
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = str(picSeged)
    for listedObject in range(0, len(picObjects)):
        object = ET.SubElement(annotation, 'object')
        object_name = ET.SubElement(object, 'name')
        object_name.text = picObjects[listedObject].name
        object_pose = ET.SubElement(object, 'pose')
        object_pose.text = 'Unspecified'
        object_truncated = ET.SubElement(object, 'truncated')
        object_truncated.text = "0"
        object_difficult = ET.SubElement(object, 'difficult')
        object_difficult.text = "0"
        object_bndbox = ET.SubElement(object, 'bndbox')
        bndbox_xmin = ET.SubElement(object_bndbox, 'xmin')
        bndbox_xmax = ET.SubElement(object_bndbox, 'xmax')
        bndbox_ymin = ET.SubElement(object_bndbox, 'ymin')
        bndbox_ymax = ET.SubElement(object_bndbox, 'ymax')
        bndbox_xmin.text = str(picObjects[listedObject].xmin)
        bndbox_xmax.text = str(picObjects[listedObject].xmax)
        bndbox_ymin.text = str(picObjects[listedObject].ymin)
        bndbox_ymax.text = str(picObjects[listedObject].ymax)
    annot_xml = ET.tostring(annotation)
    if not os.path.isdir(f"{picLocation}"):
        os.makedirs(f"{picLocation}")
    with open(f"{picLocation}/{picFilename}.xml", "wb+") as f:
        f.write(annot_xml)


def DataWriteWork(fileINDEX, sourceImgPATH, noiseImgLIST, folderPATH, ARGs , NoiseSourceList):
    original_clean, source, bw_groundTruth, color_groundTruth, source_annote_list = ImageMaker_ImageDraw(sourceImgPATH,
                                                                                                         noiseImgLIST,
                                                                                                         ARGs.obj_name,NoiseSourceList)
    CreateXMLData(picLocation=folderPATH.Annotations, picFilename=str(fileINDEX),
                  picObjects=source_annote_list, picSeged=ARGs.is_seged)

    source.save(f"{folderPATH.JPEGImages}/{str(fileINDEX)}.jpg")
    bw_groundTruth.save(f"{folderPATH.SegmentationBW}/{str(fileINDEX)}.png")
    color_groundTruth.save(f"{folderPATH.SegmentationClass}/{str(fileINDEX)}.png")
    original_clean.save(f"{folderPATH.CleanOrigin}/{str(fileINDEX)}.png")


if __name__ == "__main__":

    args = get_args()
    if args.num_object > 1:
        raise Exception("Multiple Object Generation lacks implementation and checking")

    imgDir = f"{args.data_root}/{args.raw_path}"
    imgList = glob.glob(os.path.join(imgDir, "*.png"), recursive=True) + glob.glob(os.path.join(imgDir, "*.jpg"), recursive=True)
    log.info(f"loaded img list")
    log.info(f"Total: {len(imgList)}")
    noiseDir = f"{args.data_root}/{args.noise_path}"
    noiseSourceList = glob.glob(os.path.join(noiseDir, "*.png"), recursive=True)
    log.info(f"loaded noise list")

    log.info(f"current noise ratio: {args.ratio}")

    main_Path = FolderPaths(args.data_root, args.prefix)
    if args.n2n_enabled == 1:
        log.info("N2n mode activated, creating paired folders")
        paired_Path = FolderPaths(args.data_root, f"{args.prefix}_paired")
    else:
        raise NotImplementedError("Data Gen Without N2N is not implemented!")

    parameterDictionLIST = []
    log.info(f"amount of images: {len(imgList) * args.ratio}")
    for x in range(0, len(imgList) * args.ratio):
        noiseList = []
        for y in range(0, args.num_object):
            noiseList.append(random.choice(noiseSourceList))
        parameterTUPLE = (x, imgList[x % len(imgList)], noiseList, main_Path, args, noiseSourceList)
        parameterDictionLIST.append(parameterTUPLE)

        if args.n2n_enabled == 1:
            parameterTUPLE = (x, imgList[x % len(imgList)], noiseList, paired_Path, args,noiseSourceList)
            parameterDictionLIST.append(parameterTUPLE)
    log.info(f"paramter tuple appended")
    log.info("Creating Multiprocessing Pool")
    pool = multiprocessing.Pool()
    log.info("Starting pool")
    pool.starmap(DataWriteWork, parameterDictionLIST)
