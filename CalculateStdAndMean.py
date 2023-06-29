import argparse
import logging

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from utils.n2ndataloader import NoiseRGB_2_NoiseRGB_Dataset

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--prefix', '-p', dest='prefix', default="Bodymark_Dataset", help='dataset folder prefix')
    return parser.parse_args()


args = get_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), interpolation=Image.NEAREST)])

dataset = NoiseRGB_2_NoiseRGB_Dataset(sourceDir=f'data/{args.prefix}',
                                      targetDir=f'data/{args.prefix}_paired',
                                      transform=transform, target_transform=transform)
loader = DataLoader(
    dataset,
    batch_size=512,
    num_workers=0,
    shuffle=False
)

mean = None
std = None
for data, _ in loader:
    for b in range(data.shape[0]):
        print(b)
        if mean is None and std is None:
            std, mean = torch.std_mean(data[b], dim=(1, 2))
        else:
            std = std + torch.std(data[b], dim=(1, 2))
            mean = mean + torch.mean(data[b], dim=(1, 2))
    print(f'mean:{mean / 512}, std:{std / 512}')
