from inspect import Traceback
from sampler import InfiniteSamplerWrapper
import network as net
import argparse
from datetime import time
from logging import Logger
from pathlib import Path
import time
import os
import cv2
from datasets import *
import traceback

import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
import yaml
from tqdm import tqdm
import logging
import torchvision
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


# cudnn.benchmark = True
# Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
# ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_transform(config):
    transform_list = [
        transforms.Resize(size=(config['img_size'], config['img_size'])),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def test_transfrom(config):
    transform_list = [
        transforms.Resize(size=(config['img_size'], config['img_size'])),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def adjust_learning_rate(opt, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = opt['lr'] / (1.0 + opt['lr_decay'] * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# # Basic options
parser.add_argument('--config', type=str, default='config/TrainConfig.yaml',
                    help='Config of training RPNet.')

args = parser.parse_args()
with open(args.config) as f:
    opt = yaml.load(f, Loader=yaml.Loader)

device = torch.device('cuda')

output = Path(opt['output'])

log_dir = output / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)

checkpoint_dir = output / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True, parents=True)

test_dir = output / 'test'
test_dir.mkdir(exist_ok=True, parents=True)

writer = SummaryWriter(log_dir=str(log_dir))

# decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(opt['vgg']))
vgg_relu4_1 = nn.Sequential(*list(vgg.children())[:31])

if opt['network'] == 'adain':
    network = net.AdaINRPNet(opt, vgg_relu4_1)
elif opt['network'] == 'multi_adain':
    network = net.MultiScaleAdaINRPNet(opt, vgg_relu4_1)
elif opt['network'] == 'sel_multi_adain':
    network = net.SELastMultiScaleAdaINRPNet(opt, vgg_relu4_1)
elif opt['network'] == 'wct':
    network = net.WCTRPNet(opt, vgg_relu4_1)
elif opt['network'] == 'ld_adain':
    network = net.LDMSAdaINRPNet(opt, vgg_relu4_1)
elif opt['network'] == 'ld_adain2':
    network = net.LDMSAdaINRPNet2(opt, vgg_relu4_1)
elif opt['network'] == 'dynamic_sanet':
    network = net.AdaptiveSAModel(opt, vgg, opt['start_iter'], opt['img_size'])
elif opt['network'] == 'sanet':
    network = net.SAModel(opt, vgg, opt['start_iter'], opt['img_size'])
elif opt['network'] == 'mrf':
    network = net.MRFRPNet(opt, vgg_relu4_1)
elif opt['network'] == 'spade':
    network = net.SpadeRPNet(opt, vgg_relu4_1)

network.eval()
network = network.cuda()

print(network)
test_tf = test_transfrom(opt)

if opt['test_dataset'] == 'photoreal':
    test_dataset = PhotorealisticPariedDataset(opt['test_dir'], test_tf)
elif opt['test_dataset'] == 'fmt':
    test_dataset = FmtDataset(opt['test_dir'], test_tf)
elif opt['test_dataset'] == 'paired':
    test_dataset = PariedDataset(opt['test_dir'], test_tf)

test_dataloader = data.DataLoader(
    test_dataset, batch_size=opt['batch_size'], num_workers=opt['num_workers'])

with torch.no_grad():
    for idx, (content_images, style_images, content_name, style_name, c_mask_path, s_mask_path) in enumerate(test_dataloader):
        content_images = content_images.cuda()
        style_images = style_images.cuda()
        stylizeds = network.test(content_images, style_images, iterations=i,
                                 bid=idx, c_mask_path=c_mask_path, s_mask_path=s_mask_path)
        output_dir = test_dir / 'test_output'
        output_dir.mkdir(exist_ok=True, parents=True)
        for b_idx, (content_img, style_img, stylized, cn, sn) in enumerate(zip(content_images, style_images, stylizeds, content_name, style_name)):
            images = torch.stack(
                [content_img, style_img, stylized], dim=0)
            cat_output_path = output_dir / \
                f'{cn}-{sn}-cat.png'
            stylized_output_path = output_dir / \
                f'{cn}-{sn}.png'
            torchvision.utils.save_image(
                images, cat_output_path, nrow=3)
            torchvision.utils.save_image(
                stylized, stylized_output_path, nrow=1)
            logger.info(f'Proceed {cn}-{sn}.')
