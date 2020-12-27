from numpy.lib.arraypad import pad
from torch import strided
import torch.nn as nn
from torch.nn.modules import padding
import torch
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss




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
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)






class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
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

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
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
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = AdaIN(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.rp_decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s