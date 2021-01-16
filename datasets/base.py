import torch.utils.data as data
from pathlib import Path
from PIL import Image
import os

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

class PariedDataset(data.Dataset):
    def __init__(self, root, transform):
        super(PariedDataset, self).__init__()
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

        style_name = self.content_names[index]

        style_path = os.path.join(self.style_dir, style_name)

        content_img = Image.open(str(content_path)).convert('RGB')
        style_img = Image.open(str(style_path)).convert('RGB')

        content_img = self.transform(content_img)
        style_img = self.transform(style_img)

        return content_img, style_img, os.path.splitext(os.path.basename(str(content_path)))[0], os.path.splitext(os.path.basename(str(style_path)))[0],[],[]

    def __len__(self):
        return len(self.content_names)

    def name(self):
        return 'PairedDataset'


class PhotorealisticPariedDataset(data.Dataset):
    def __init__(self, root, transform):
        super(PhotorealisticPariedDataset, self).__init__()
        self.root = root
        self.content_dir = os.path.join(self.root, 'content')
        self.style_dir = os.path.join(self.root, 'style')
        self.seg_dir = os.path.join(self.root, 'labelme_segmentation')

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

        c_name = os.path.splitext(os.path.basename(str(content_path)))[0]
        s_name = os.path.splitext(os.path.basename(str(style_path)))[0]

        c_mask_path =  os.path.join(self.seg_dir, f'{c_name}.png')
        s_mask_path = os.path.join(self.seg_dir, f'{s_name}.png')
        return content_img, style_img, c_name, s_name,c_mask_path, s_mask_path

    def __len__(self):
        return len(self.content_names)

    def name(self):
        return 'PairedDataset'


class FmtDataset(data.Dataset):
    def __init__(self, root, transform, fmt='*'):
        super(FmtDataset, self).__init__()
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
        return 'FmtDataset'
