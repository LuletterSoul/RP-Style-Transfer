from sampler import InfiniteSamplerWrapper
import network as net
import argparse
from datetime import time
from logging import Logger
from pathlib import Path
import time
import os
import cv2

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


def train_transform():
    transform_list = [
        transforms.RandomResizedCrop(512),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def test_transfrom():
    transform_list = [
        # transforms.RandomCrop(256),
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform, fmt='*/P*', root2=None):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob(fmt))
        if root2 is not None:
            path2 = list(Path(root2).glob('*'))
            print(path2[0])
            self.paths.extend(path2)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


class Dataset(data.Dataset):
    def __init__(self, root, transform, fmt='*'):
        super(Dataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob(fmt))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

class CityspacesDataset(data.Dataset):
    def __init__(self, img_dir,transform, img_size=256):
        super(CityspacesDataset, self).__init__()
        self.img_dir = self.img_dir
        self.img_names = os.listdir(self.img_dir)
        self.transform = transform
        self.img_size = 256

        ignore_label = -1

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path = os.path.join(
            self.img_dir, self.img_names[index])
        # img = Image.open(img_path).convert('RGB')
    
        img = cv2.imread(img_path)

        content = img[:,:self.img_size,:] 
        label = img[:,self.img_size: self.img_size * 2, :]

        content = Image.fromarray(cv2.cvtColor(content, cv2.COLOR_BGR2RGB))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = self.convert_label(label)
        label = Image.fromarray(label)

        content = self.transform(content)

        return content, label
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __len__(self):
        return len(self.img_names)

    def name(self):
        return 'PairedDataset'
    

class PairedDataset(data.Dataset):
    def __init__(self, root, transform):
        super(PairedDataset, self).__init__()
        self.root = root
        self.content_dir = os.path.join(self.root, 'content')
        self.style_dir = os.path.join(self.root, 'style')

        self.content_names = os.listdir(self.content_dir)
        self.style_names = os.listdir(self.style_dir)
        # self.content_paths = list(Path(self.root).glob('content/*'))
        # self.style_paths = list(Path(self.root).glob('style/*'))
        # self.paths = list(zip(self.content_paths, self.style_paths))
        self.transform = transform

    def __getitem__(self, index):

        content_path = os.path.join(
            self.content_dir, self.content_names[index])

        style_name = 'tar{}'.format(
            self.content_names[index].replace('in', ''))

        style_path = os.path.join(self.style_dir, style_name)

        content_img = Image.open(str(content_path)).convert('RGB')
        style_img = Image.open(str(style_path)).convert('RGB')

        content_img = self.transform(content_img)
        style_img = self.transform(style_img)

        return content_img, style_img, os.path.splitext(os.path.basename(str(content_path)))[0], os.path.splitext(os.path.basename(str(style_path)))[0]

    def __len__(self):
        return len(self.content_names)

    def name(self):
        return 'PairedDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = opt.lr / (1.0 + opt.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# # Basic options
parser.add_argument('--config', type=str, default='config/TrainConfig.yaml',
                    help='Config of training RPNet.')
# parser.add_argument('--content_dir', type=str, required=True,
# help='Directory path to a batch of content images')

# parser.add_argument('--content_dir_2', type=str, default=None,
#                     help='Directory path to a batch of content images')

# parser.add_argument('--test_dir', type=str, default=None,
#                     help='Directory path to pairs of test images.')

# parser.add_argument('--style_dir', type=str, required=True,

#                     help='Directory path to a batch of style images')
# parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# # training options
# parser.add_argument('--output', default='./experiments',
#                     help='Directory to save the model')
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--lr_decay', type=float, default=5e-5)
# parser.add_argument('--max_iter', type=int, default=160000)
# parser.add_argument('--batch_size', type=int, default=8)
# parser.add_argument('--style_weight', type=float, default=10.0)
# parser.add_argument('--content_weight', type=float, default=1.0)
# parser.add_argument('--n_threads', type=int, default=16)
# parser.add_argument('--n_threads', type=int, default=16)
# parser.add_argument('--save_model_interval', type=int, default=10000)
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
vgg = nn.Sequential(*list(vgg.children())[:31])

if opt['network'] == 'adain':
    network = net.AdaINRPNet(opt, vgg)
elif opt['network'] == 'mrf':
    network = net.MRFRPNet(opt, vgg)
elif opt['network'] == 'spade':
    network = net.SpadeRPNet(opt, vgg)

network.train()
network = network.cuda()
print(network)

content_tf = train_transform()
style_tf = train_transform()

test_tf = test_transfrom()

# content_dataset = FlatFolderDataset(
# args.content_dir, content_tf, root2=args.content_dir_2)
# style_dataset = FlatFolderDataset(args.style_dir, style_tf, '*/C*')
content_dataset = Dataset(
    opt['content_dir'], content_tf)
style_dataset = Dataset(
    opt['style_dir'], style_tf, fmt='*/*')
test_dataset = PairedDataset(opt['test_dir'], test_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=opt['batch_size'],
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=opt['num_workers']))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=opt['batch_size'],
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=opt['num_workers']))

test_dataloader = data.DataLoader(
    test_dataset, batch_size=opt['batch_size'], num_workers=opt['num_workers'])

optimizer = torch.optim.Adam(network.parameters(), lr=opt['lr'])

for i in range(1, opt['max_iter']):

    start = time.time()
    optimizer.zero_grad()
    # adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).cuda().detach()
    style_images = next(style_iter).cuda().detach()
    # loss_mrf, loss_c, loss_s = network(content_images, style_images)
    loss_dict, total_loss = network(content_images, style_images)
    # loss = loss_c + loss_s + loss_mrf

    total_loss.backward()
    optimizer.step()

    end = time.time()

    eclipse_time = round(end - start, 2)
    loss_str = ''
    for key, loss_item in loss_dict.items():
        writer.add_scalar(key, loss_item, i)
        loss_str += f', {key} {loss_item}'

    if i % opt['test_iter'] == 0:
        for idx, (content_images, style_images, content_name, style_name) in enumerate(test_dataloader):
            content_images = content_images.cuda()
            style_images = style_images.cuda()
            stylizeds = network.test(content_images, style_images)
            output_dir = test_dir / f'{i}'
            output_dir.mkdir(exist_ok=True, parents=True)
            for b_idx, (content_img, style_img, stylized, cn, sn) in enumerate(zip(content_images, style_images, stylizeds, content_name, style_name)):
                images = torch.stack([content_img, style_img, stylized], dim=0)
                output_path = output_dir / \
                    f'{cn}-{sn}.png'
                torchvision.utils.save_image(
                    images, output_path, nrow=3)
                logger.info(f'Proceed {cn}-{sn}.')

    if i % opt['log_iter'] == 0:
        logger.info(
            f'Iterations {i}, elapsed time: {eclipse_time} {loss_str}')

    if i % opt['snapshot_save_iter'] == 0 or (i + 1) == opt['max_iter']:
        torch.save(network.state_dict(), checkpoint_dir /
                   f'{i+1}.pth')
writer.close()
