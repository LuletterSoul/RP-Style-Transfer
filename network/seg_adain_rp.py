from .base import *


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
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

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


def build_increase_depth_rp_blocks(block_num, in_dim, hidden_dim, out_dim, ks=3, stride=1, pd=1,activation='lrelu'):
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

    

class SegRPNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.seg_head = 


class SegAdaINRPNet(nn.Module):
    def __init__(self, config, vgg_encoder) -> None:
        super().__init__()
        self.adain_rp_net = AdaINRPNet(config, vgg_encoder)
        self.seg_rp_net = 