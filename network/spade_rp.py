
from .base import *

def build_spade_rp_blocks(block_num, in_dim, hidden_dim, out_dim, ks=3, stride=1, pd=1):
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

class SPADE(nn.Module):
    def __init__(self, param_free_norm_type, norm_nc, condition_nc):
        super().__init__()

        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        # ks = int(parsed.group(2))
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(
                norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(condition_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(
            nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, conditional):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        conditional = F.interpolate(
            conditional, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(conditional)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SpadeResnetBlock(nn.Module):
    def __init__(self, fin, fout, spade_norm, condition_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        # spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_norm, fin, condition_nc)
        self.norm_1 = SPADE(spade_norm, fmiddle, condition_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_norm, fin, condition_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, condition):
        x_s = self.shortcut(x, condition)

        dx = self.conv_0(self.actvn(self.norm_0(x, condition)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, condition)))

        out = x_s + dx

        return out

    def shortcut(self, x, condition):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, condition))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SpadeDecoder(nn.Module):
    def __init__(self, ndf, spade_norm, condition_nc) -> None:
        super(SpadeDecoder, self).__init__()

        self.head = SpadeResnetBlock(
            condition_nc, 16 * ndf, spade_norm, condition_nc)

        self.rp_middle_0 = SpadeResnetBlock(
            16 * ndf, 16 * ndf, spade_norm, condition_nc)
        self.rp_middle_1 = SpadeResnetBlock(
            16 * ndf, 16 * ndf, spade_norm, condition_nc)

        self.d1 = SpadeResnetBlock(16 * ndf, 8 * ndf, spade_norm, condition_nc)
        self.d2 = SpadeResnetBlock(8 * ndf, 4 * ndf, spade_norm, condition_nc)
        self.d3 = SpadeResnetBlock(4 * ndf, 2 * ndf, spade_norm, condition_nc)
        self.d4 = SpadeResnetBlock(2 * ndf, 1 * ndf, spade_norm, condition_nc)

        final_nc = ndf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

    def forward(self, feat, condition):

        feat = self.head(feat, condition)

        feat = self.rp_middle_0(feat, condition)

        feat = self.rp_middle_1(feat, condition)

        feat = self.d1(feat, condition)
        feat = self.d2(feat, condition)
        feat = self.d3(feat, condition)
        feat = self.d4(feat, condition)

        img = self.conv_img(feat)

        return img


class SpadeRPNet(nn.Module):
    def __init__(self, config, vgg_encoder) -> None:
        super(SpadeRPNet, self).__init__()
        # super(Net, self).__init__()
        enc_layers = list(vgg_encoder.children())
        self.config = config
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        assert self.config['rp_blocks'] - 2 >= 0

        # build resolution perserving encoder, which is composed of blocks with increasing depth.
        self.encoder_out_dim = self.config['hidden_dim'] * \
            2 ** (self.config['rp_blocks'] - 1)

        self.rp_content_encoder = build_increase_depth_rp_blocks(
            self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim)

        self.rp_style_encoder = build_increase_depth_rp_blocks(
            self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim)

        # build resolution perserving decoder, which is composed of blocks with decreasing depth.
        self.decoder_in_dim = self.encoder_out_dim
        self.decoder_hidden_dim = self.decoder_in_dim // 2
        # self.rp_decoder = build_decrease_depth_rp_blocks(
        # self.config['rp_blocks'], self.decoder_in_dim, self.decoder_hidden_dim, 3)
        self.rp_decoder = SpadeDecoder(
            self.config['ndf'], self.config['spade_norm'], self.decoder_in_dim)

        self.loss_mrf = MRFLoss(self.config['k'])
        self.mse_loss = MSELoss()

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
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    def test(self, content, style):
        with torch.no_grad():
            content_feat = self.rp_content_encoder(content)
            style_feat = self.rp_style_encoder(style)
            stylized = self.rp_decoder(style_feat, content_feat)
            return stylized

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        # content_feat = self.encode(content)
        content_feat = self.rp_content_encoder(content)
        style_feat = self.rp_style_encoder(style)

        # fusion_feat = AdaIN(content_feat, style_feat)

        # stylized = self.rp_decoder(fusion_feat)
        stylized = self.rp_decoder(style_feat, content_feat)

        down_stylized_feats = self.encode_with_intermediate(stylized)
        down_style_feats = self.encode_with_intermediate(style)
        down_content_feats = self.encode_with_intermediate(content)

        loss_s = self.calc_style_loss(
            down_stylized_feats[0], down_style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(
                down_stylized_feats[i], down_style_feats[i])
        loss_c = self.calc_content_loss(
            down_stylized_feats[-1], down_content_feats[-1])

        total_loss = self.config['content_weight'] * \
            loss_c + self.config['style_weight'] * loss_s
        return {
            'style_loss': loss_s,
            'content_loss': loss_c,
            'total_loss': total_loss
        }, total_loss
