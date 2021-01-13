from .base import *
from .adain_rp import SegAdaINRPNet
from .base import adaptive_instance_normalization as AdaIN


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



    

class SegRPNet(nn.Module):
    def __init__(self, config, encoder_out_dim) -> None:
        super().__init__()
        self.config = config
        self.seg_head = build_rp_blocks(self.config['rp_blocks'], encoder_out_dim , self.config['seg_hidden_dim'],self.config['class_num'])

    def forward(self, x):
        return self.seg_head(x)


class SegAdaINRPNet(nn.Module):
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
        
        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
        #                         1.0166, 0.9969, 0.9754, 1.0489,
        #                         0.8786, 1.0023, 0.9539, 0.9843, 
        #                         1.1116, 0.9037, 1.0865, 1.0955, 
        #                         1.0865, 1.1529, 1.0507]).cuda()
        
        # self.seg_rp_net = SegRPNet(config,self.encoder_out_dim)

        self.mse_loss = MSELoss()
        self.ce_loss = CrossEntropy(ignore_label=-1,weight=self.class_weights)

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
            content_feat = self.rp_shared_encoder(content)
            style_feat = self.rp_shared_encoder(style)
            fusion_feat = AdaIN(content_feat, style_feat)
            stylized = self.rp_decoder(fusion_feat)
            return stylized

    def forward(self, content, style,content_label, style_label, alpha=1.0):
        assert 0 <= alpha <= 1
        # content_feat = self.encode(content)
        content_feat = self.rp_shared_encoder(content)
        style_feat = self.rp_shared_encoder(style)

        # content_pred = self.seg_rp_net(content_feat)
        # style_pred = self.seg_rp_net(style_feat)

        # content_label = torch.argmax(content_pred, dim =1)
        # label_label = torch.argmax(style_pred, dim =1)

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



    
