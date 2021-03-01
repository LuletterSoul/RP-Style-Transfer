from abc import abstractmethod
from numpy.lib.arraypad import pad
import numpy as np
from torch import strided
import torch.nn as nn
from torch.nn.modules import padding
import torch
from torch.nn.modules.container import ModuleList, Sequential
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
import torchvision
from PIL import Image
from .attention import *

torchvision.models.inception


class StackType(object):
    Deeper = 'deeper'
    Shallower = 'shallower'
    Constant = 'constant'
    DShallower = 'dec_shallower'


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='lrelu', pad_type='reflect', inception_num=None, attention=None):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = nn.SpectralNorm(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim,
                                  kernel_size, stride, bias=self.use_bias)
        if inception_num:
            self.inception = ModuleList()
            for i in range(inception_num):
                self.inception.append(nn.Sequential(nn.Conv2d(output_dim, output_dim,
                                                              1, 1, bias=self.use_bias)))
            self.inception = nn.Sequential(*self.inception)
        else:
            self.inception = None

        if attention == 'se':
            self.attention_block = SEBottleneck(
                inplanes=output_dim, planes=output_dim)
        elif attention == 'sk':
            self.attention_block = SKBottleneck(
                inplanes=output_dim, planes=output_dim)
        else:
            self.attention_block = None
        self.attention_map = None

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.inception:
            x = self.inception(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.attention_block:
            x = self.attention_block(x)
            self.attention_map = self.attention_block.attention_map
        return x


def build_rp_blocks(block_num, in_dim, hidden_dim, out_dim, ks=3, stride=1, pd=1, activation='lrelu'):
    rp_blocks = ModuleList()

    rp_blocks.append(Conv2dBlock(input_dim=in_dim,
                                 output_dim=hidden_dim,
                                 kernel_size=ks,
                                 stride=stride,
                                 padding=pd,
                                 activation=activation))

    for i in range(0, block_num-2):
        rp_blocks.append(Conv2dBlock(input_dim=hidden_dim,
                                     output_dim=hidden_dim,
                                     kernel_size=ks,
                                     stride=stride,
                                     padding=pd,
                                     activation=activation))

        hidden_dim *= 2

    rp_blocks.append(Conv2dBlock(input_dim=hidden_dim,
                                 output_dim=out_dim,
                                 kernel_size=ks,
                                 stride=stride,
                                 padding=pd,
                                 activation=activation))

    return nn.Sequential(*rp_blocks)


def rp_deeper_conv_blocks(block_num, in_dim, hidden_dim, out_dim, ks=3, stride=1, pd=1, activation='lrelu', inception_num=None):
    rp_blocks = ModuleList()

    rp_blocks.append(Conv2dBlock(input_dim=in_dim,
                                 output_dim=hidden_dim,
                                 kernel_size=ks,
                                 stride=stride,
                                 padding=pd,
                                 activation=activation, inception_num=inception_num))

    for i in range(0, block_num-2):
        rp_blocks.append(Conv2dBlock(input_dim=hidden_dim,
                                     output_dim=hidden_dim * 2,
                                     kernel_size=ks,
                                     stride=stride,
                                     padding=pd,
                                     activation=activation, inception_num=inception_num))
        hidden_dim *= 2

    rp_blocks.append(Conv2dBlock(input_dim=hidden_dim,
                                 output_dim=out_dim,
                                 kernel_size=ks,
                                 stride=stride,
                                 padding=pd,
                                 activation=activation, inception_num=inception_num))

    return rp_blocks


def rp_constant_conv_blocks(block_num, in_dim, hidden_dim, out_dim, ks=3, stride=1, pd=1, activation='lrelu', inception_num=None, attention=False):
    rp_blocks = ModuleList()

    rp_blocks.append(Conv2dBlock(input_dim=in_dim,
                                 output_dim=hidden_dim,
                                 kernel_size=ks,
                                 stride=stride,
                                 padding=pd,
                                 activation=activation, inception_num=inception_num, attention=attention))

    for i in range(0, block_num-2):
        rp_blocks.append(Conv2dBlock(input_dim=hidden_dim,
                                     output_dim=hidden_dim,
                                     kernel_size=ks,
                                     stride=stride,
                                     padding=pd,
                                     activation=activation, inception_num=inception_num, attention=attention))

    rp_blocks.append(Conv2dBlock(input_dim=hidden_dim,
                                 output_dim=out_dim,
                                 kernel_size=ks,
                                 stride=stride,
                                 padding=pd,
                                 activation=activation, inception_num=inception_num, attention=attention))

    return rp_blocks


def rp_shallower_conv_blocks(block_num, in_dim, hidden_dim, out_dim, ks=3, stride=1, pd=1, activation='lrelu', incread_depth=True):
    rp_blocks = ModuleList()

    rp_blocks.append(Conv2dBlock(input_dim=in_dim,
                                 output_dim=hidden_dim,
                                 kernel_size=ks,
                                 stride=stride,
                                 padding=pd,
                                 activation=activation))

    for i in range(0, block_num-2):
        rp_blocks.append(Conv2dBlock(input_dim=hidden_dim,
                                     output_dim=hidden_dim // 2,
                                     kernel_size=ks,
                                     stride=stride,
                                     padding=pd,
                                     activation=activation))
        hidden_dim //= 2

    rp_blocks.append(Conv2dBlock(input_dim=hidden_dim,
                                 output_dim=out_dim,
                                 kernel_size=ks,
                                 stride=stride,
                                 padding=pd,
                                 activation=activation))

    return rp_blocks


def cal_affinity_map(content_feat, style_feat, k=3, reverse=False, c_mask=None, s_mask=None):
    assert content_feat.size() == style_feat.size()
    N, C, H, W = content_feat.size()
    content_feat = content_feat.squeeze()
    style_feat = style_feat.squeeze()
    n_content_feat = F.normalize(content_feat, dim=0).view(C, -1)
    n_style_feat = F.normalize(style_feat, dim=0).view(C, -1)
    attention_map = torch.mm(n_content_feat.t(), n_style_feat)
    mask = torch.ones(H * W, H * W, dtype=n_content_feat.dtype).cuda()
    if reverse:
        attention_map *= -1

    # if c_mask is not None and s_mask is not None:
    #     c_mask = F.interpolate(c_mask.float(), [H, W]).long().view(H * W, 1)
    #     s_mask = F.interpolate(s_mask.float(), [H, W]).long().view(1, H * W)
    #     # test = (c_mask == s_mask).long()
    #     # print(torch.sum(test))
    #     mask = (c_mask == s_mask).type_as(n_content_feat)

    attention_map *= mask
    affinity_map = torch.zeros(H * W, H * W, dtype=n_content_feat.dtype).cuda()
    index = torch.topk(attention_map, k, 0)[1]
    value = torch.ones(k, H * W, dtype=n_content_feat.dtype).cuda()
    affinity_map.scatter_(0, index, value)  # set weight matrix

    index = torch.topk(attention_map, k, 1)[1]
    value = torch.ones(H * W, k, dtype=n_content_feat.dtype).cuda()
    affinity_map.scatter_(1, index, value)  # set weight matrix

    return affinity_map


def cal_dist(A, B):
    """
    :param A: (d, m) m个d维向量
    :param B: (d, n) n个d维向量
    :return: (m, n)
    """
    ASize = A.size()
    BSize = B.size()
    dist = (torch.sum(A ** 2, dim=0).view(ASize[1], 1).expand(ASize[1], BSize[1]) +
            torch.sum(B ** 2, dim=0).view(1, BSize[1]).expand(ASize[1], BSize[1]) -
            2 * torch.mm(A.t(), B)).to(A.device)
    return dist


def build_increase_depth_rp_blocks(block_num, in_dim, hidden_dim, out_dim, ks=3, stride=1, pd=1):

    rp_blocks = ModuleList()
    rp_blocks.append(
        nn.Conv2d(in_dim, hidden_dim, kernel_size=ks, stride=stride, padding=pd))
    rp_blocks.append(nn.ReLU(inplace=True))

    for i in range(0, block_num-2):
        rp_blocks.append(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=ks, stride=stride, padding=pd))
        rp_blocks.append(nn.ReLU(inplace=True))
        hidden_dim *= 2

    rp_blocks.append(nn.Conv2d(hidden_dim, out_dim,
                               kernel_size=ks, padding=pd))
    rp_blocks.append(nn.ReLU(inplace=True))
    return nn.Sequential(*rp_blocks)


def build_decrease_depth_rp_blocks(block_num, in_dim, hidden_dim, out_dim, ks=3, stride=1, pd=1):
    rp_blocks = ModuleList()
    rp_blocks.append(
        nn.Conv2d(in_dim, hidden_dim, kernel_size=ks, stride=stride, padding=pd))
    rp_blocks.append(nn.ReLU(inplace=True))
    for i in range(0, block_num-2):
        rp_blocks.append(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=ks, stride=stride, padding=pd))
        rp_blocks.append(nn.ReLU(inplace=True))
        hidden_dim //= 2

    rp_blocks.append(nn.Conv2d(hidden_dim, out_dim,
                               kernel_size=ks, padding=pd))
    rp_blocks.append(nn.ReLU(inplace=True))
    return nn.Sequential(*rp_blocks)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def compute_label_info(content_segment, style_segment):
    if not content_segment.size or not style_segment.size:
        return None, None
    max_label = np.max(content_segment) + 1
    label_set = np.unique(content_segment)
    label_indicator = np.zeros(max_label)
    for l in label_set:
        content_mask = np.where(content_segment.reshape(
            content_segment.shape[0] * content_segment.shape[1]) == l)
        style_mask = np.where(style_segment.reshape(
            style_segment.shape[0] * style_segment.shape[1]) == l)

        c_size = content_mask[0].size
        s_size = style_mask[0].size
        if c_size > 10 and s_size > 10 and c_size / s_size < 100 and s_size / c_size < 100:
            label_indicator[l] = True
        else:
            label_indicator[l] = False
    return label_set, label_indicator


def get_segment_and_info(content_seg_path, style_seg_path, content_shape, style_shape):
    """
    :param content_seg_path:
    :param style_seg_path:
    :param content_shape: (wc, hc)
    :param style_shape: (ws, hs)
    :return:
    """
    c_seg = np.asarray(Image.open(content_seg_path).resize(content_shape))
    s_seg = np.asarray(Image.open(style_seg_path).resize(style_shape))
    print(
        f'content_seg_label={np.unique(c_seg)}， style_seg_label={np.unique(s_seg)}')
    label_set, label_indicator = compute_label_info(c_seg, s_seg)
    return c_seg, s_seg, label_set, label_indicator


def get_index(feat, label, device):
    mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
    if mask[0].size <= 0:
        return None
    return torch.LongTensor(mask[0]).to(device)


def calc_mean_std_for_masked_feat(masked_feat, eps=1e-5):
    """
    :param masked_feat: (c, k)
    :param eps:
    :return: ()
    """
    c, k = masked_feat.size()
    feat_var = masked_feat.var(dim=1) + eps
    feat_std = feat_var.sqrt().view(c, 1)
    feat_mean = masked_feat.mean(dim=1).view(c, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization_for_masked_feat(masked_content_feat, masked_style_feat):
    """
    :param masked_content_feat: (c, nc)
    :param masked_style_feat: (c, ns)
    :return: (c, nc)
    """
    size = masked_content_feat.size()
    content_mean, content_std = calc_mean_std_for_masked_feat(
        masked_content_feat)
    style_mean, style_std = calc_mean_std_for_masked_feat(masked_style_feat)
    # print(f'masked_content_feat.size={masked_content_feat.size()}, content_mean.size={content_mean.size()}')
    normalized_feat = (masked_content_feat -
                       content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adaptive_instance_normalization_with_segment(content_feat, style_feat, content_seg_path, style_seg_path):
    """
    :param content_feat: (1, c, hc, wc)
    :param style_feat:  (1, c, hs, ws)
    :param content_seg_path: content segment path
    :param style_seg_path: style segment path
    :return:
    """
    device = content_feat.device
    # TODO add size judge
    content_shape = (content_feat.size()[3], content_feat.size()[2])
    style_shape = (style_feat.size()[3], style_feat.size()[2])
    c_seg, s_seg, label_set, label_indicator = get_segment_and_info(content_seg_path, style_seg_path, content_shape,
                                                                    style_shape)
    content_feat_size = content_feat.size()
    style_feat_size = style_feat.size()
    content_feat = content_feat.squeeze(0).view(content_feat_size[1], -1)
    style_feat = style_feat.squeeze(0).view(style_feat_size[1], -1)
    target_feat = content_feat.clone()
    for label in label_set:
        if not label_indicator[label]:
            # invalid label
            continue
        content_index = get_index(c_seg, label, device)
        style_index = get_index(s_seg, label, device)
        if content_index is None or style_index is None:
            continue
        masked_content_feat = torch.index_select(
            content_feat, dim=1, index=content_index)
        masked_style_feat = torch.index_select(
            style_feat, dim=1, index=style_index)
        normalized_feat = adaptive_instance_normalization_for_masked_feat(
            masked_content_feat, masked_style_feat)
        target_feat.index_copy_(1, content_index, normalized_feat)
    target_feat = target_feat.view(
        content_feat_size[1], content_feat_size[2], content_feat_size[3]).unsqueeze(0)
    return target_feat


class BaseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.begin = 0

    @abstractmethod
    def test(self, content, style, iterations=0, bid=0, c_mask_path=None, s_mask_path=None):
        pass

    @abstractmethod
    def save():
        pass

    @abstractmethod
    def decode(self, content_feats, style_feats, use_mask=False, c_mask_path=None, s_mask_path=None):
        pass

    @abstractmethod
    def fuse(self, content_feats, style_feats):
        pass

    @abstractmethod
    def encode_with_intermediate(self, input):
        pass

    def save(self, save_path, iterations=0):
        torch.save(self.state_dict(), save_path)


class SourceNet(BaseNet):
    def __init__(self, config, vgg_encoder):
        super(BaseNet, self).__init__()
        enc_layers = list(vgg_encoder.children())
        self.config = config
        self.begin = 0
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def test(self, content, style, iterations=0, bid=0, c_mask_path=None, s_mask_path=None):
        with torch.no_grad():
            content_feats = self.encode_with_intermediate(content)
            style_feats = self.encode_with_intermediate(style)
            stylized = self.decode(
                content_feats, style_feats, self.config['use_mask'], c_mask_path, s_mask_path)
            return stylized

    def decode(self, content_feats, style_feats, use_mask=False, c_mask_path=None, s_mask_path=None):
        t = self.do_mask_stylized(
            content_feats[-1], style_feats[-1], c_mask_path, s_mask_path) if use_mask else adaptive_instance_normalization(content_feats[-1], style_feats[-1])
        # t = self.fuse(content_feats, style_feats)
        g_t = self.decoder(t)
        return g_t

    def do_mask_stylized(self, content_feat, style_feat, c_mask_path, s_mask_path):
        mask_stylized = []
        for bid, (cf, sf) in enumerate(zip(content_feat, style_feat)):
            mask_stylized.append(adaptive_instance_normalization_with_segment(cf.unsqueeze(0), sf.unsqueeze(
                0), c_mask_path[bid], s_mask_path[bid]))
        stylized = torch.cat(mask_stylized, dim=0)
        return stylized

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        content_feats = self.encode_with_intermediate(content)
        style_feats = self.encode_with_intermediate(style)

        t = adaptive_instance_normalization(content_feats[-1], style_feats[-1])
        g_t = self.decode(content_feats, style_feats)

        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        total_loss = self.config['content_weight'] * \
            loss_c + self.config['style_weight'] * loss_s
        return {
            'style_loss': loss_s,
            'content_loss': loss_c,
            'total_loss': total_loss
        }, total_loss
