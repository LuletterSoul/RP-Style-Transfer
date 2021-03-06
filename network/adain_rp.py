from torch import add, stack, symeig
from .base import *
from .base import adaptive_instance_normalization as AdaIN
from .base import adaptive_instance_normalization_with_segment as AdaINSeg
import os
import seaborn
import random
import matplotlib.pyplot as plt

from utils.common import make_grid
from utils.mst import *
from torchvision import transforms


class AdaINRPNet(BaseNet):
    def __init__(self, config, vgg_encoder) -> None:
        super(AdaINRPNet, self).__init__()
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

        # dim = self.config['out_dim']

        # rp_blocks = ModuleList()
        # rp_blocks.append(
        #     nn.Conv2d(self.config['in_dim'], dim, kernel_size=3, padding=1))
        # rp_blocks.append(nn.ReLU(inplace=True))

        # for i in range(1, self.config['rp_blocks']):
        #     rp_blocks.append(
        #         nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1))
        #     rp_blocks.append(nn.ReLU(inplace=True))
        #     dim *= 2

        # self.enc_out_dim = dim

        assert self.config['rp_blocks'] - 2 >= 0

        # build resolution perserving encoder, which is composed of blocks with increasing depth.
        self.encoder_out_dim = self.config['hidden_dim'] * \
            2 ** (self.config['rp_blocks'] - 1)

        self.rp_shared_encoder = build_increase_depth_rp_blocks(
            self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim)

        # self.rp_style_encoder = build_increase_depth_rp_blocks(
        #    self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim)

        # build resolution perserving decoder, which is composed of blocks with decreasing depth.
        self.decoder_in_dim = self.encoder_out_dim
        self.decoder_hidden_dim = self.decoder_in_dim // 2
        self.rp_decoder = build_decrease_depth_rp_blocks(
            self.config['rp_blocks'], self.decoder_in_dim, self.decoder_hidden_dim, 3)

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

    def fuse(self, content_feats, style_feats):
        fusion = AdaIN(content_feats, style_feats)
        return fusion

    def test(self, content, style, iterations=0, bid=0, c_mask_path=None, s_mask_path=None):
        with torch.no_grad():
            content_feat = self.rp_shared_encoder(content)
            style_feat = self.rp_shared_encoder(style)
            # fusion_feat = AdaIN(content_feat, style_feat)
            fusion_feat = self.fuse(content_feat, style_feat)
            stylized = self.rp_decoder(fusion_feat)
            return stylized

    def save(self, save_path, iterations=0):
        state_dict = {
            'encoder': self.rp_shared_encoder.state_dict(),
            'decoder': self.rp_decoder.state_dict()
        }
        torch.save(state_dict, save_path)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        # content_feat = self.encode(content)
        content_feat = self.rp_shared_encoder(content)
        style_feat = self.rp_shared_encoder(style)

        fusion_feat = AdaIN(content_feat, style_feat)

        stylized = self.rp_decoder(fusion_feat)

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


class MultiScaleAdaINRPNet(AdaINRPNet):
    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)
        self.config = config
        self.rp_shared_encoder = None
        self.rp_decoder = None
        self._shuffle = self.config['shuffle']
        self._shuffle_layers = self.config['shuffle_layers']
        self._sort = self.config['sort']
        self.layer_num = self.config['rp_blocks']
        self._stylized_layers = self.config['stylized_layers']
        if self.config['enc_stack_way'] == StackType.Deeper:
            self.rp_shared_encoder = rp_deeper_conv_blocks(
                self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim, inception_num=self.config['inception_num'])
            self.rp_decoder = rp_shallower_conv_blocks(
                self.config['rp_blocks'], self.decoder_in_dim, self.decoder_hidden_dim, 3)

        elif self.config['enc_stack_way'] == StackType.Constant:
            self.encoder_out_dim = self.config['hidden_dim']
            self.rp_shared_encoder = rp_constant_conv_blocks(
                self.config['rp_blocks'],
                3,
                self.config['hidden_dim'],
                self.encoder_out_dim, inception_num=self.config['inception_num'],
                attention=self.config['attention'])
            self.decoder_in_dim = self.encoder_out_dim
            self.rp_decoder = rp_constant_conv_blocks(
                self.config['rp_blocks'], self.decoder_in_dim, self.config['hidden_dim'], 3)

        # elif self.config['enc_stack_way'] == StackType.DShallower:
        #     self.encoder_out_dim = self.config['hidden_dim']
        #     self.rp_shared_encoder = rp_constant_conv_blocks(
        #         self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim, inception_num=self.config['inception_num'], se=self.config['se'])
        #     self.decoder_in_dim = self.encoder_out_dim
        #     self.rp_decoder = rp_constant_conv_blocks(
        #         self.config['rp_blocks'], self.decoder_in_dim, self.config['hidden_dim'], 3)

        if self.config['resume']:
            checkpoint_path = self.config['checkpoint_path']
            self.begin = int(os.path.splitext(
                os.path.basename(checkpoint_path))[0])
            state_dict = torch.load(self.config['checkpoint_path'])
            self.rp_shared_encoder.load_state_dict(state_dict['encoder'])
            self.rp_decoder.load_state_dict(state_dict['decoder'])
            print(f'Loaded checkpoint from {checkpoint_path}')

    def encode_rp_intermediate(self, input):
        results = [input]
        for i in range(len(self.rp_shared_encoder)):
            results.append(self.rp_shared_encoder[i](results[-1]))
        return results[1:]

    def visualize_channel_attention(self, bid, iterations=0):
        """visualize channel attention map, batch size is fixed at 1
        """
        attentions = [enc.attention_map for enc in self.rp_shared_encoder]
        fig, ax = plt.subplots(
            self.config['rp_blocks'], 1, constrained_layout=True)
        for idx, attention in enumerate(attentions):
            b, c, _, _ = attention.size()
            attention_map = attention.detach().view(
                b * c, 1).permute(1, 0).cpu().numpy()
            ax[idx].set_title(f'Layer {idx}')  # 设置x轴图例为空值
            seaborn.heatmap(data=attention_map,
                            vmin=0, vmax=1, ax=ax[idx])
            clamp_output = os.path.join(self.config['output'], 'claim_map')
            if not os.path.exists(clamp_output):
                os.makedirs(clamp_output, exist_ok=True)
            plt.show()
            plt.savefig(os.path.join(
                clamp_output, f'it_{iterations}_bid_{bid}.png'))
        plt.clf()
        plt.close()

    def visualize_feature_map(self, bid, iterations, reference_imgs, feats, suffix='content'):
        layer_num = self.config['rp_blocks']
        unloader = transforms.ToPILImage()
        reference_imgs = unloader(reference_imgs.squeeze().detach().cpu())
        sample_feats = [f[:, :8, :, :] for f in feats]
        sample_feats = torch.cat(sample_feats, dim=1).squeeze()
        sample_feats = [unloader(f.detach().cpu()) for f in sample_feats]
        visualize_img = make_grid(
            reference_imgs, sample_feats, 8, unit_size=256)
        visualize_output = os.path.join(self.config['output'], 'visualize')
        if not os.path.exists(visualize_output):
            os.makedirs(visualize_output, exist_ok=True)
        visualize_img.save(os.path.join(
            visualize_output, f'it_{iterations}_bid_{bid}_{suffix}.png'))

    def sort_by_weights(self, feats):
        """
           sort feature maps across channel dimension by attention weights

        Args:
            feats ([type]): [description]
        """
        attentions = [enc.attention_map for enc in self.rp_shared_encoder]
        sorted_feats = []
        for idx, attention in enumerate(attentions):
            feat = feats[idx]
            sort_feat = []
            _, indexes = attention.sort(dim=1, descending=True)
            for bid, index in enumerate(indexes):
                index = index.view(-1)
                sort_feat.append(torch.index_select(
                    feat[bid].unsqueeze(0), dim=1, index=index))
            sort_feat = torch.cat(sort_feat, dim=0)
            sorted_feats.append(sort_feat)
        return sorted_feats

    def test(self, content, style, iterations=0, bid=0, c_mask_path=None, s_mask_path=None):
        self.eval()
        with torch.no_grad():
            content_feats = self.encode_rp_intermediate(content)
            style_feats = self.encode_rp_intermediate(style)
            if self._shuffle:
                content_feats = [self.shuffle(c, idx)
                                 for idx, c in enumerate(content_feats)]
                style_feats = [self.shuffle(s, idx)
                               for idx, s in enumerate(style_feats)]
            stylized = self.decode(
                content_feats, style_feats, use_mask=self.config['use_mask'], c_mask_path=c_mask_path, s_mask_path=s_mask_path)
            # self.visualize_channel_attention(bid, iterations)
            # self.visualize_feature_map(
            # bid, iterations, content, content_feats, suffix='content')
            # self.visualize_feature_map(
            # bid, iterations, style, style_feats, suffix='style')
            self.train()
            return stylized

    # def decode(self, content_feats, style_feats, use_mask=False, c_mask_path=None, s_mask_path=None):
    #     stylized = AdaIN(content_feats[-1], style_feats[-1])
    #     stylized = self.rp_decoder[0](stylized)
    #     for i, (content_feat, style_feat) in enumerate(list(zip(content_feats[:-1], style_feats[:-1]))[::-1]):
    #         if use_mask:
    #             mask_stylized = []
    #             for bid, (cf, sf) in enumerate(zip(content_feat, style_feat)):
    #                 mask_stylized.append(AdaINSeg(cf.unsqueeze(0), sf.unsqueeze(
    #                     0), c_mask_path[bid], s_mask_path[bid]))
    #             stylized = torch.cat(mask_stylized, dim=0)
    #         else:
    #             stylized = AdaIN(stylized, style_feat)
    #         stylized = self.rp_decoder[i+1](stylized)
    #     return stylized

    def decode(self, content_feats, style_feats, use_mask=False, c_mask_path=None, s_mask_path=None):
        # if self._shuffle:
        # content_feats = [self.shuffle(c) for c in content_feats]
        if self._sort:
            content_feats = self.sort_by_weights(content_feats)
            style_feats = self.sort_by_weights(style_feats)
        stylized = self.do_mask_stylized(
            content_feats[-1], style_feats[-1], c_mask_path, s_mask_path) if use_mask else AdaIN(content_feats[-1], style_feats[-1])
        stylized = self.rp_decoder[0](stylized)
        for i, (content_feat, style_feat) in enumerate(list(zip(content_feats[:-1], style_feats[:-1]))[::-1]):
            if use_mask:
                fusion_stylized = self.do_mask_stylized(
                    content_feat, style_feat, c_mask_path, s_mask_path)
            else:
                fusion_stylized = AdaIN(content_feat, style_feat)
            stylized = self.rp_decoder[i+1](stylized + fusion_stylized)
        return stylized

    def shuffle(self, feats, layer):
        if layer > self._shuffle_layers:
            return feats
        N, C, H, W = feats.size()
        groups = 4
        out = feats.view(N, groups, C // groups, H, W).permute(0,
                                                               2, 1, 3, 4).contiguous().view(N, C, H, W)
        return out

    def do_mask_stylized(self, content_feat, style_feat, c_mask_path, s_mask_path):
        mask_stylized = []
        for bid, (cf, sf) in enumerate(zip(content_feat, style_feat)):
            mask_stylized.append(AdaINSeg(cf.unsqueeze(0), sf.unsqueeze(
                0), c_mask_path[bid], s_mask_path[bid]))
        stylized = torch.cat(mask_stylized, dim=0)
        return stylized

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        # content_feat = self.encode(content)
        content_feats = self.encode_rp_intermediate(content)
        style_feats = self.encode_rp_intermediate(style)
        stylized = self.decode(content_feats, style_feats)
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


class CCAMDec(nn.Module):
    """
    CCAM decoding module
    """

    def __init__(self):
        super(CCAMDec, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1)).cuda()

    def forward(self, x, y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,H,W)
            returns :
                out : compact channel attention feature
                attention map: K*C
        """
        x = x.detach()
        y = y.detach()
        m_batchsize, C, width, height = x.size()
        x_reshape = x.view(m_batchsize, C, -1)

        B, K, W, H = y.size()
        y_reshape = y.view(B, K, -1)
        proj_query = x_reshape  # BXC1XN
        proj_key = y_reshape.permute(0, 2, 1)  # BX(N)XC
        energy = torch.bmm(proj_query, proj_key)  # BXC1XC
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B, K, -1)  # BCN

        out = torch.bmm(attention, proj_value)  # BC1N
        out = out.view(m_batchsize, C, width, height)

        out = x + self.scale*out
        return out


class CrossChannelAttentionRPNet(MultiScaleAdaINRPNet):
    """embedded cross channel attention between content features and style features.

    Args:
        MultiScaleAdaINRPNet ([type]): [description]
    """

    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)
        self.channel_attentions = [CCAMDec()
                                   for i in range(self.layer_num)]

    def decode(self, content_feats, style_feats, use_mask=False, c_mask_path=None, s_mask_path=None):
        # if self._shuffle:
        # content_feats = [self.shuffle(c) for c in content_feats]
        stylized = self.do_mask_stylized(
            content_feats[-1], style_feats[-1], c_mask_path, s_mask_path) if use_mask else AdaIN(content_feats[-1], style_feats[-1])
        attention_res = self.channel_attentions[0](
            content_feats[-1], style_feats[-1])
        stylized = self.rp_decoder[0](stylized + attention_res)
        for i, (content_feat, style_feat) in enumerate(list(zip(content_feats[:-1], style_feats[:-1]))[::-1]):
            if i + 1 < self._stylized_layers:  # control the stylized layers
                if use_mask:
                    stylized = self.do_mask_stylized(
                        stylized, style_feat, c_mask_path, s_mask_path)
                else:
                    stylized = AdaIN(stylized, style_feat)
                # use channel attention to enhance feature response.
                attention_res = self.channel_attentions[i +
                                                        1](stylized, style_feat)
                stylized = self.rp_decoder[i +
                                           1](stylized + attention_res)
            else:
                stylized = self.rp_decoder[i+1](stylized)
        return stylized


class GlobalMSTRPNet(MultiScaleAdaINRPNet):
    """embedded graph cut algorithm for matching content and style channels.

    Args:
        MultiScaleAdaINRPNet ([type]): [description]
    """

    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)
        self.mst = MultimodalStyleTransfer(3, 1, 'cuda:0', 0, None)

    def decode(self, content_feats, style_feats, use_mask=False, c_mask_path=None, s_mask_path=None):
        # if self._shuffle:
        # content_feats = [self.shuffle(c) for c in content_feats]
        stylized = self.mst.transfer(
            content_feats[-1].detach(), style_feats[-1].detach())
        stylized = self.rp_decoder[0](stylized)
        for i, (content_feat, style_feat) in enumerate(list(zip(content_feats[:-1], style_feats[:-1]))[::-1]):
            if i + 1 < self._stylized_layers:  # control the stylized layers
                stylized = self.rp_decoder[i +
                                           1](self.mst.transfer(stylized, style_feat.detach()))
            else:
                stylized = self.rp_decoder[i+1](stylized)
        return stylized


class SELastMultiScaleAdaINRPNet(MultiScaleAdaINRPNet):
    """use se attention after Adain fusion

    Args:
        MultiScaleAdaINRPNet ([type]): [description]
    """

    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)
        self.attention_block = SEBottleneck(
            inplanes=self.config['hidden_dim'], planes=self.config['hidden_dim'])

    def decode(self, content_feats, style_feats, use_mask=False, c_mask_path=None, s_mask_path=None):
        stylized = AdaIN(content_feats[-1], style_feats[-1])
        stylized = self.rp_decoder[0](stylized)
        reverse_feat_pairs = list(
            zip(content_feats[:-1], style_feats[:-1]))[::-1]
        for i, (content_feat, style_feat) in enumerate(reverse_feat_pairs):
            if use_mask:
                mask_stylized = []
                for bid, (cf, sf) in enumerate(zip(content_feat, style_feat)):
                    mask_stylized.append(AdaINSeg(cf.unsqueeze(0), sf.unsqueeze(
                        0), c_mask_path[bid], s_mask_path[bid]))
                stylized = torch.cat(mask_stylized, dim=0)
            else:
                stylized = AdaIN(stylized, style_feat)
                if i == len(reverse_feat_pairs) - 1:
                    # adopts se attention on fusion features for content and style
                    stylized = self.attention_block(stylized)
            stylized = self.rp_decoder[i+1](stylized)
        return stylized


class LDMSAdaINRPNet(MultiScaleAdaINRPNet):
    """use 7*7 conv to aggregate more spatial information.

    Args:
        MultiScaleAdaINRPNet ([type]): [description]
    """

    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)
        self.hidden_dim = self.config['hidden_dim']
        self.config = config
        self.inception_num = config['inception_num']
        self.layer_num = config['ld_layer_num']
        self.stylized_layers = config['stylized_layers']
        self.build_encoders()
        self.build_decoders()

    def build_encoders(self):

        hidden_dim = self.hidden_dim
        setattr(self, 'rp_enc0_small_revf', Conv2dBlock(
            3, hidden_dim, 3, 1, 1, inception_num=self.inception_num))
        setattr(self, 'rp_enc0_big_revf', Conv2dBlock(
            3, hidden_dim, 3, 1, 1, inception_num=self.inception_num))

        for i in range(self.layer_num-1):
            hidden_dim *= 2
            setattr(self, f'rp_enc{i+1}_small_revf', Conv2dBlock(hidden_dim,
                                                                 hidden_dim, 3, 1, 1, inception_num=self.inception_num))
            setattr(self, f'rp_enc{i+1}_big_revf', Conv2dBlock(hidden_dim,
                                                               hidden_dim, 7, 1, 3, inception_num=self.inception_num))
        self.encoder_out_dim = hidden_dim

    def build_decoders(self):
        hidden_dim = self.encoder_out_dim
        for i in range(self.layer_num-1):
            if i < self.stylized_layers - 1:
                setattr(self, f'rp_dec{i}', Conv2dBlock(
                    hidden_dim * 2, hidden_dim, 3, 1, 1, inception_num=self.inception_num))
            elif i == self.stylized_layers - 1:
                setattr(self, f'rp_dec{i}', Conv2dBlock(
                    hidden_dim * 2, hidden_dim // 2, 3, 1, 1, inception_num=self.inception_num))
            else:
                setattr(self, f'rp_dec{i}', Conv2dBlock(
                    hidden_dim, hidden_dim // 2, 3, 1, 1, inception_num=self.inception_num))
            hidden_dim //= 2

        if self.stylized_layers >= self.layer_num:
            setattr(self, f'rp_dec{self.layer_num-1}', Conv2dBlock(
                hidden_dim * 2, 3, 3, 1, 1, inception_num=self.inception_num))
        else:
            setattr(self, f'rp_dec{self.layer_num-1}', Conv2dBlock(
                hidden_dim, 3, 3, 1, 1, inception_num=self.inception_num))

    def decode(self, content_feats, style_feats, use_mask=False, c_mask_path=None, s_mask_path=None):
        stylized = self.do_mask_stylized(
            content_feats[-1], style_feats[-1], c_mask_path, s_mask_path) if use_mask else AdaIN(content_feats[-1], style_feats[-1])
        stylized = self.rp_dec0(stylized)
        for i, (content_feat, style_feat) in enumerate(list(zip(content_feats[:-1], style_feats[:-1]))[::-1]):
            fusion_stylized = []
            if i < self.stylized_layers-1:  # control stylization layers
                # do multiscale stylization
                if use_mask:
                    fusion_stylized = self.do_mask_stylized(
                        content_feat, style_feat, c_mask_path, s_mask_path)
                else:
                    fusion_stylized = AdaIN(stylized, style_feat)
            stylized = getattr(
                self, f'rp_dec{i+1}')(stylized + fusion_stylized)
        return stylized

    def encode_rp_intermediate(self, input):
        results = [input]
        for i in range(self.layer_num):
            rp_enc_small_revf_feat = getattr(
                self, f'rp_enc{i}_small_revf')(results[-1])
            rp_enc_big_revf_feat = getattr(
                self, f'rp_enc{i}_big_revf')(results[-1])
            results.append(
                torch.cat([rp_enc_small_revf_feat, rp_enc_big_revf_feat], dim=1))
        return results[1:]

    def save(self, save_path, iterations=0):
        torch.save(self.state_dict(), save_path)


class LDMSAdaINRPNet2(LDMSAdaINRPNet):
    """use pooling branch to extract more high-level features

    Args:
        LDMSAdaINRPNet ([type]): [description]
    """

    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)

    def build_encoders(self):

        hidden_dim = self.hidden_dim

        setattr(self, 'rp_enc0_small_revf', Conv2dBlock(
            3, hidden_dim, 3, 1, 1, inception_num=self.inception_num))
        setattr(self, 'rp_enc0_big_revf', nn.Sequential(
            nn.Conv2d(3, hidden_dim, (1, 1)), nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1))))

        for i in range(self.layer_num-1):
            hidden_dim *= 2
            setattr(self, f'rp_enc{i+1}_small_revf', Conv2dBlock(hidden_dim,
                                                                 hidden_dim, 3, 1, 1, inception_num=self.inception_num))
            setattr(self, f'rp_enc{i+1}_big_revf', nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, (1, 1)
                          ), nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
                nn.ReLU(),  # relu1-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
                nn.ReLU(),  # relu1-2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1))))

        self.encoder_out_dim = hidden_dim

    def encode_rp_intermediate(self, input):
        results = [input]
        for i in range(self.layer_num):
            rp_enc_small_revf_feat = getattr(
                self, f'rp_enc{i}_small_revf')(results[-1])
            rp_enc_big_revf_feat = getattr(
                self, f'rp_enc{i}_big_revf')(results[-1])
            size = rp_enc_small_revf_feat.size()[2:]
            rp_enc_big_revf_feat = nn.functional.interpolate(
                rp_enc_big_revf_feat, size)
            results.append(
                torch.cat([rp_enc_small_revf_feat, rp_enc_big_revf_feat], dim=1))
        return results[1:]


class LDMSAdaINRPNet3(LDMSAdaINRPNet2):
    """use two encoders to extract different level features.
       feature transformation is built on element-wise sum.

    Args:
        MultiScaleAdaINRPNet ([type]): [description]
    """

    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)

    def build_encoders(self):

        hidden_dim = self.hidden_dim

        setattr(self, 'rp_enc0_small_revf', Conv2dBlock(
            3, hidden_dim, 3, 1, 1, inception_num=self.inception_num))
        setattr(self, 'rp_enc0_big_revf', nn.Sequential(
            nn.Conv2d(3, hidden_dim, (1, 1)), nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1))))

        for i in range(self.layer_num-1):
            setattr(self, f'rp_enc{i+1}_small_revf', Conv2dBlock(hidden_dim,
                                                                 hidden_dim, 3, 1, 1, inception_num=self.inception_num))
            setattr(self, f'rp_enc{i+1}_big_revf', nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, (1, 1)
                          ), nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
                nn.ReLU(),  # relu1-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
                nn.ReLU(),  # relu1-2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1))))
        self.encoder_out_dim = hidden_dim

    def build_decoders(self):
        hidden_dim = self.encoder_out_dim
        for i in range(self.layer_num-1):
            if i < self.stylized_layers - 1:
                setattr(self, f'rp_dec{i}', Conv2dBlock(
                    hidden_dim * 2, hidden_dim * 2, 3, 1, 1, inception_num=self.inception_num))
            elif i == self.stylized_layers - 1:
                setattr(self, f'rp_dec{i}', Conv2dBlock(
                    hidden_dim * 2, hidden_dim, 3, 1, 1, inception_num=self.inception_num))
            else:
                setattr(self, f'rp_dec{i}', Conv2dBlock(
                    hidden_dim, hidden_dim, 3, 1, 1, inception_num=self.inception_num))

        if self.stylized_layers >= self.layer_num:
            setattr(self, f'rp_dec{self.layer_num-1}', Conv2dBlock(
                hidden_dim * 2, 3, 3, 1, 1, inception_num=self.inception_num))
        else:
            setattr(self, f'rp_dec{self.layer_num-1}', Conv2dBlock(
                hidden_dim, 3, 3, 1, 1, inception_num=self.inception_num))

    def encode_rp_intermediate(self, input):
        fine_feats = [input]
        coarse_feats = [input]
        fusion_feats = []
        for i in range(self.layer_num):
            rp_enc_small_revf_feat = getattr(
                self, f'rp_enc{i}_small_revf')(fine_feats[-1])
            rp_enc_big_revf_feat = getattr(
                self, f'rp_enc{i}_big_revf')(coarse_feats[-1])
            fine_feats.append(rp_enc_small_revf_feat)
            coarse_feats.append(rp_enc_big_revf_feat)

            size = rp_enc_small_revf_feat.size()[2:]
            rp_enc_big_revf_feat = nn.functional.interpolate(
                rp_enc_big_revf_feat, size)
            fusion_feats.append(
                torch.cat([rp_enc_small_revf_feat, rp_enc_big_revf_feat], dim=1))

        return fusion_feats


class LDMSAdaINRPNet4(LDMSAdaINRPNet3):
    """use two encoders to extract different level features.
       feature transformation is built on channel-wise concatenation.

    Args:
        MultiScaleAdaINRPNet ([type]): [description]
    """

    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)

    def build_encoders(self):

        hidden_dim = self.hidden_dim

        setattr(self, 'rp_enc0_small_revf', Conv2dBlock(
            3, hidden_dim, 3, 1, 1, inception_num=self.inception_num))
        setattr(self, 'rp_enc0_big_revf', nn.Sequential(
            nn.Conv2d(3, hidden_dim, (1, 1)), nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)))

        for i in range(self.layer_num-1):
            setattr(self, f'rp_enc{i+1}_small_revf', Conv2dBlock(hidden_dim,
                                                                 hidden_dim, 3, 1, 1, inception_num=self.inception_num))
            setattr(self, f'rp_enc{i+1}_big_revf', nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, (1, 1)
                          ), nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
                nn.ReLU(),  # relu1-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(hidden_dim, hidden_dim, (3, 3)),
                nn.ReLU(),  # relu1-2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)))
        self.encoder_out_dim = hidden_dim

    def build_decoders(self):
        hidden_dim = self.encoder_out_dim
        addition = 0
        for i in range(self.layer_num):  # 0,1,2,3
            if i < self.stylized_layers - 1:
                setattr(self, f'rp_dec{i}', Conv2dBlock(
                    addition + hidden_dim * 2, hidden_dim * 2, 3, 1, 1, inception_num=self.inception_num))
                addition = hidden_dim * 2
            elif i == self.stylized_layers - 1:
                setattr(self, f'rp_dec{i}', Conv2dBlock(
                    addition + hidden_dim * 2, hidden_dim, 3, 1, 1, inception_num=self.inception_num))
                addition = hidden_dim * 2
            else:
                setattr(self, f'rp_dec{i}', Conv2dBlock(
                    hidden_dim + addition, hidden_dim, 3, 1, 1, inception_num=self.inception_num))

        # layernum=5, stylized_layers = 5
        # 0, 64 64
        # 1, 128 64
        # 2, 128 64
        # 3, 128 64
        # 4, 128 64
        if self.stylized_layers == self.layer_num:
            setattr(self, f'rp_dec{self.layer_num-1}', Conv2dBlock(
                addition + hidden_dim * 2, 3, 3, 1, 1, inception_num=self.inception_num))
        else:
            setattr(self, f'rp_dec{self.layer_num-1}', Conv2dBlock(
                hidden_dim + addition, 3, 3, 1, 1, inception_num=self.inception_num))

    def decode(self, content_feats, style_feats, use_mask=False, c_mask_path=None, s_mask_path=None):
        # stylized = AdaIN(content_feats[-1], style_feats[-1])
        stylized = self.do_mask_stylized(
            content_feats[-1], style_feats[-1], c_mask_path, s_mask_path) if use_mask else AdaIN(content_feats[-1], style_feats[-1])
        stylized = getattr(
            self, f'rp_dec0')(stylized)
        for i, (content_feat, style_feat) in enumerate(list(zip(content_feats[:-1], style_feats[:-1]))[::-1]):
            if use_mask:
                prefix_stylized = self.do_mask_stylized(
                    content_feat, style_feat, c_mask_path, s_mask_path)
            else:
                prefix_stylized = AdaIN(content_feat, style_feat)
            # channel-wise concatenation
            fusion_stylized = torch.cat([stylized, prefix_stylized], dim=1)
            # print(stylized.size())
            # print(prefix_stylized.size())
            # print(fusion_stylized.size())
            stylized = getattr(
                self, f'rp_dec{i+1}')(fusion_stylized)
        return stylized

    def encode_rp_intermediate(self, input):
        fine_feats = [input]
        coarse_feats = [input]
        fusion_feats = []
        for i in range(self.layer_num):
            rp_enc_small_revf_feat = getattr(
                self, f'rp_enc{i}_small_revf')(fine_feats[-1])
            rp_enc_big_revf_feat = getattr(
                self, f'rp_enc{i}_big_revf')(coarse_feats[-1])
            fine_feats.append(rp_enc_small_revf_feat)
            coarse_feats.append(rp_enc_big_revf_feat)

            size = rp_enc_small_revf_feat.size()[2:]
            rp_enc_big_revf_feat = nn.functional.interpolate(
                rp_enc_big_revf_feat, size)
            fusion_feats.append(
                torch.cat([rp_enc_small_revf_feat, rp_enc_big_revf_feat], dim=1))

        return fusion_feats


class LDMSAdaINRPNet5(LDMSAdaINRPNet4):
    """use two encoders to extract different level features.
       feature transformation is built on channel-wise concatenation.
       using transpose2d to align spatial dimensions.

    Args:
        MultiScaleAdaINRPNet ([type]): [description]
    """

    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)
        self.ups = ModuleList()
        in_size = 256
        stride = 2
        for i in range(self.layer_num):
            self.ups.append(nn.ConvTranspose2d(
                self.hidden_dim, self.hidden_dim, kernel_size=2 ** (i+1), stride=2 ** (i+1)))

    def encode_rp_intermediate(self, input):
        fine_feats = [input]
        coarse_feats = [input]
        fusion_feats = []
        for i in range(self.layer_num):
            rp_enc_small_revf_feat = getattr(
                self, f'rp_enc{i}_small_revf')(fine_feats[-1])
            rp_enc_big_revf_feat = getattr(
                self, f'rp_enc{i}_big_revf')(coarse_feats[-1])
            fine_feats.append(rp_enc_small_revf_feat)
            coarse_feats.append(rp_enc_big_revf_feat)
            # size = rp_enc_small_revf_feat.size()[2:]
            # rp_enc_big_revf_feat = nn.functional.interpolate(
            # rp_enc_big_revf_feat, size)
            rp_enc_big_revf_feat = self.ups[i](rp_enc_big_revf_feat)
            fusion_feats.append(
                torch.cat([rp_enc_small_revf_feat, rp_enc_big_revf_feat], dim=1))

        return fusion_feats
