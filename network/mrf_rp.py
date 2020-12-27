from .base import *


class MRFLoss(nn.Module):

    def __init__(self, k, mask=None, mean='mean') -> None:
        super().__init__()
        self.mask = mask
        self.k = k
        self.mean = mean

    def forward(self, content_feat, style_feat):
        N, C, H, W = content_feat.size()
        flat_content_feat = content_feat.view(C, -1)
        flat_style_feat = style_feat.view(C, -1)
        dist_matrix = cal_dist(flat_content_feat, flat_style_feat)
        attention_map = cal_affinity_map(content_feat, style_feat, self.k)
        assert attention_map.size() == dist_matrix.size()
        weighted_dist_matrix = attention_map * dist_matrix
        if self.mean == 'mean':
            return weighted_dist_matrix.sum() / (H * W * self.k)
        else:
            return weighted_dist_matrix.mean()

class MRFRPNet(nn.Module):
    def __init__(self, config, vgg_encoder) -> None:
        super(MRFRPNet, self).__init__()
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

        self.rp_content_encoder = build_increase_depth_rp_blocks(
            self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim)

        self.rp_style_encoder = build_increase_depth_rp_blocks(
            self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim)

        # build resolution perserving decoder, which is composed of blocks with decreasing depth.
        self.decoder_in_dim = self.encoder_out_dim * 2
        self.decoder_hidden_dim = self.decoder_in_dim // 2
        self.rp_decoder = build_decrease_depth_rp_blocks(
            self.config['rp_blocks'], self.decoder_in_dim, self.decoder_hidden_dim, 3)

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
            cat_feat = torch.cat([content_feat, style_feat], dim=1)
            stylized = self.rp_decoder(cat_feat)
            return stylized

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        # content_feat = self.encode(content)
        content_feat = self.rp_content_encoder(content)
        style_feat = self.rp_style_encoder(style)

        fusion_feat = torch.cat([content_feat, style_feat], dim=1)

        stylized = self.rp_decoder(fusion_feat)

        down_stylized_feats = self.encode_with_intermediate(stylized)
        down_style_feats = self.encode_with_intermediate(style)

        content_feat_prime = self.rp_content_encoder(stylized)
        style_feat_prime = self.rp_style_encoder(stylized)
        loss_mrf = self.loss_mrf(
            down_stylized_feats[-1], down_style_feats[-1])
        loss_s = self.calc_style_loss(style_feat_prime, style_feat)
        loss_c = self.calc_content_loss(content_feat_prime, content_feat)
        total_loss = self.config['content_weight'] * \
            loss_c + self.config['style_weight'] * \
            loss_s + self.config['mrf_weight'] * loss_mrf
        return {
            'mrf_loss': loss_mrf,
            'style_loss': loss_s,
            'content_loss': loss_c,
            'total_loss': total_loss
        }, total_loss


