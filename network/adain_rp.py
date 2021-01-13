from .base import *
from .base import adaptive_instance_normalization as AdaIN


class SegAdaINRPNet(BaseNet):
    def __init__(self, config, vgg_encoder) -> None:
        super(SegAdaINRPNet, self).__init__()
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

    def test(self, content, style,iterations=0,bid=0):
        with torch.no_grad():
            content_feat = self.rp_shared_encoder(content)
            style_feat = self.rp_shared_encoder(style)
            fusion_feat = AdaIN(content_feat, style_feat)
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


class MultiScaleAdaINRPNet(SegAdaINRPNet):
    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config,vgg_encoder)
        self.config = config
        if self.config['enc_stack_way'] == StackType.Deeper:
           self.rp_shared_encoder = rp_deeper_conv_blocks(
                            self.config['rp_blocks'], 3, self.config['hidden_dim'], inception_num=self.config['inception_num'])
           self.rp_decoder = rp_shallower_conv_blocks(
                self.config['rp_blocks'], self.decoder_in_dim, self.decoder_hidden_dim, 3)

        elif self.config['enc_stack_way'] == StackType.Constant:
            self.encoder_out_dim = self.config['hidden_dim']
            self.rp_shared_encoder =rp_constant_conv_blocks(
                self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim,inception_num=self.config['inception_num'])
            self.decoder_in_dim = self.encoder_out_dim
            self.rp_decoder = rp_constant_conv_blocks(
                    self.config['rp_blocks'], self.decoder_in_dim, self.config['hidden_dim'], 3)
    
    def encode_rp_intermediate(self, input):
        results = [input]
        for i in range(len(self.rp_shared_encoder)):
            results.append(self.rp_shared_encoder[i](results[-1]))
        return results[1:]
    
    def test(self, content, style, iterations=0,bid=0):
        self.eval()
        with torch.no_grad():
            content_feats = self.encode_rp_intermediate(content)
            style_feats = self.encode_rp_intermediate(style)
            stylized= AdaIN(content_feats[-1], style_feats[-1])
            stylized = self.decode(content_feats,style_feats)
            self.train()
            return stylized
    
    def decode(self,content_feats, style_feats):
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
        #     print(style_feat.size())

        # for content_feat in content_feats:
        #     print(content_feat.size())
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


class LDMSAdaINRPNet(MultiScaleAdaINRPNet):
    def __init__(self, config, vgg_encoder) -> None:
        super().__init__(config, vgg_encoder)
        hidden_dim = 8

        self.layer_num = 5

        setattr(self,'rp_enc0_small_revf',Conv2dBlock(3,hidden_dim,3,1,1,inception_num=1))
        setattr(self,'rp_enc0_big_revf',Conv2dBlock(3,hidden_dim,3,1,1,inception_num=1))

        for i in range(self.layer_num-1):
            hidden_dim *=2
            setattr(self,f'rp_enc{i+1}_small_revf',Conv2dBlock(hidden_dim,hidden_dim,3,1,1,inception_num=1))
            setattr(self,f'rp_enc{i+1}_big_revf',Conv2dBlock(hidden_dim,hidden_dim,7,1,3,inception_num=1))
    
        for i in range(self.layer_num-1):
            setattr(self,f'rp_dec{i}',Conv2dBlock(hidden_dim * 2,hidden_dim,3,1,1,inception_num=1))
            setattr(self,f'rp_dec{i}',Conv2dBlock(hidden_dim * 2,hidden_dim,3,1,1,inception_num=1))
            hidden_dim //= 2
        
        setattr(self,f'rp_dec{self.layer_num-1}',Conv2dBlock(hidden_dim * 2,3,3,1,1,inception_num=1))
        setattr(self,f'rp_dec{self.layer_num-1}',Conv2dBlock(hidden_dim * 2,3,3,1,1,inception_num=1))

    def decode(self,content_feats, style_feats):
        stylized= AdaIN(content_feats[-1], style_feats[-1])
        stylized = self.rp_dec0(stylized)
        for i, (content_feat, style_feat) in enumerate(list(zip(content_feats[:-1], style_feats[:-1]))[::-1]):
            stylized = AdaIN(stylized, style_feat)
            stylized = getattr(self,f'rp_dec{i+1}')(stylized)
        return stylized   
    
    
    def encode_rp_intermediate(self, input):
        results = [input]
        for i in range(self.layer_num):
            rp_enc_small_revf_feat = getattr(self, f'rp_enc{i}_small_revf')(results[-1])
            rp_enc_big_revf_feat = getattr(self, f'rp_enc{i}_big_revf')(results[-1])
            results.append(torch.cat([rp_enc_small_revf_feat,rp_enc_big_revf_feat],dim=1))
        return results[1:]
