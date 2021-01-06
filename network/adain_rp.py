from .base import *
from .base import adaptive_instance_normalization as AdaIN


class AdaINRPNet(nn.Module):
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

        #self.rp_style_encoder = build_increase_depth_rp_blocks(
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

    def test(self, content, style,iterations=0):
        with torch.no_grad():
            content_feat = self.rp_shared_encoder(content)
            style_feat = self.rp_style_encoder(style)
            fusion_feat = AdaIN(content_feat, style_feat)
            stylized = self.rp_decoder(fusion_feat)
            return stylized

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
        super().__init__(config,vgg_encoder)
        self.rp_shared_encoder =rp_deeper_conv_blocks(
            self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim) 
        self.rp_decoder = rp_shallower_conv_blocks(
            self.config['rp_blocks'], self.decoder_in_dim, self.decoder_hidden_dim, 3)
    
    def encode_rp_intermediate(self, input):
        results = [input]
        for i in range(len(self.rp_shared_encoder)):
            results.append(self.rp_shared_encoder[i](results[-1]))
        return results[1:]
    
    def test(self, content, style):
        with torch.no_grad():
            content_feats = self.encode_rp_intermediate(content)
            style_feats = self.encode_rp_intermediate(style)
            stylized= AdaIN(content_feats[-1], style_feats[-1])
            stylized = self.rp_decoder[0](stylized)
            for i, (content_feat, style_feat) in enumerate(list(zip(content_feats[:-1], style_feats[:-1]))[::-1]):
                stylized = AdaIN(stylized, style_feat)
                stylized = self.rp_decoder[i+1](stylized)
            return stylized

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        # content_feat = self.encode(content)
        content_feats = self.encode_rp_intermediate(content)
        style_feats = self.encode_rp_intermediate(style)
        # for style_feat in style_feats:
            # print(style_feat.size())

        stylized= AdaIN(content_feats[-1], style_feats[-1])
        stylized = self.rp_decoder[0](stylized)
        for i, (content_feat, style_feat) in enumerate(list(zip(content_feats[:-1], style_feats[:-1]))[::-1]):
            stylized = AdaIN(stylized, style_feat)
            stylized = self.rp_decoder[i+1](stylized)

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