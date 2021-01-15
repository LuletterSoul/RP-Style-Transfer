from .base import *
from .base import adaptive_instance_normalization as AdaIN
from .adain_rp import AdaINRPNet



def matrix_inv_sqrt(A):
    A = A.clone()
    a_diag_ = A.diagonal()
    a_diag_ += 1e-4
    k_c = A.shape[-1]
    c_u, c_e, c_v = torch.svd(A, some=False)

    for i in range(k_c):
        if c_e[i] < 0.00001:
            k_c = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    result = torch.mm(step1, (c_v[:, 0:k_c].t()))
    return result

def matrix_sqrt(A):
    A = A.clone()
    a_diag_ = A.diagonal()
    a_diag_ += 1e-4

    s_u, s_e, s_v = torch.svd(A, some=False)

    k_s = A.shape[-1]
    for i in range(k_s):
        if s_e[i] < 0.00001:
            k_s = i
            break

    s_d = (s_e[0:k_s]).pow(0.5)
    step1 = torch.mm(s_v[:, 0:k_s], torch.diag(s_d))
    result = torch.mm(step1, (s_v[:, 0:k_s].t()))
    return result

class WCTRPNet(BaseNet):
    def __init__(self, config, vgg_encoder) -> None:
        super(WCTRPNet, self).__init__()
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

        self.rp_shared_encoder = build_increase_depth_rp_blocks(
            self.config['rp_blocks'], 3, self.config['hidden_dim'], self.encoder_out_dim)

        if self.config['resume']:
            self.rp_shared_encoder.load_state_dict(torch.load(self.config['checkpoint_path'])['encoder'])
            checkpont_path = self.config['checkpoint_path']
            print(f'Loaded checkpoint from {checkpont_path}')
            for param in self.rp_shared_encoder.parameters():
                param.requires_grad = False

        # build resolution perserving decoder, which is composed of blocks with decreasing depth.
        self.decoder_in_dim = self.encoder_out_dim
        self.decoder_hidden_dim = self.decoder_in_dim // 2
        self.rp_decoder = build_decrease_depth_rp_blocks(
            self.config['rp_blocks'], self.decoder_in_dim, self.decoder_hidden_dim, 3)

        self.mse_loss = MSELoss()

    def whiten_and_color(self, cF, sF, method='closed-form'):
        cFSize = cF.size()
        # print(f'cF.shape = {cF.shape}')
        c_mean = torch.mean(cF, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double().to(cF.device)

        sFSize = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)

        if method == 'original':  # the original WCT by Li et al.
            cF_inv_sqrt = matrix_inv_sqrt(contentConv)
            sF_sqrt = matrix_sqrt(styleConv)
            # whiten_cF = torch.mm(cF_inv_sqrt, cF)
            # targetFeature = torch.mm(sF_sqrt,whiten_cF)
            targetFeature = sF_sqrt @ (cF_inv_sqrt @ cF)
        else:  # Lu et al.
            assert method == 'closed-form'
            cF_sqrt = matrix_sqrt(contentConv)
            cF_inv_sqrt = matrix_inv_sqrt(contentConv)
            # print(f'cF_sqrt.shape = {cF_sqrt.shape}')
            middle_matrix = matrix_sqrt(cF_sqrt @ styleConv @ cF_sqrt)
            # print(f'middle_matrix.shape = {middle_matrix.shape}')
            transform_matrix = cF_inv_sqrt @ middle_matrix @ cF_inv_sqrt
            targetFeature = transform_matrix @ cF
            # print(f'targetFeature.shape = {targetFeature.shape}')

        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

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

    def test(self, content, style,iterations=0,bid=0,c_mask_path=None,s_mask_path=None):
        self.eval()
        with torch.no_grad():
            content_feat = self.rp_shared_encoder(content)
            style_feat = self.rp_shared_encoder(style)
            fusion_feat = self.fuse(content_feat, style_feat)
            stylized = self.rp_decoder(fusion_feat)
            self.train()
            return stylized


    def save(self, save_path, iterations=0):
        state_dict = {
            'encoder': self.rp_shared_encoder.state_dict(),
            'decoder': self.rp_decoder.state_dict()
        }
        torch.save(state_dict, save_path)

    def fuse(self,content_feats, style_feats):
        fusion_feat = []
        for idx,(cf,sf) in enumerate(zip(content_feats,style_feats)):
            c,h,w = cf.size()
            cf = cf.view(c,-1).double().detach()
            sf = sf.view(c,-1).double().detach()
            wct_fs = self.whiten_and_color(cf,sf).view(c,h,w).float()
            fusion_feat.append(wct_fs)
        fusion_feat = torch.stack(fusion_feat,dim=0)
        return fusion_feat

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        # content_feat = self.encode(content)
        content_feat = self.rp_shared_encoder(content)
        style_feat = self.rp_shared_encoder(style)
        fusion_feat = self.fuse(content_feat, style_feat)
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

