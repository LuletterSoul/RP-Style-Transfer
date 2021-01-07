from genericpath import exists
from .base import *
from torch.nn import functional
import matplotlib.pyplot as plt
import seaborn 
import os

def cal_affinity_matrix(content_feat, style_feat):

    assert content_feat.size() == style_feat.size()
    b,c,h,w = content_feat.size()
    norm_content_feat = functional.normalize(content_feat.view(b,c,h*w),dim=1)
    norm_style_feat = functional.normalize(style_feat.view(b,c,h*w),dim=1)
    return torch.bmm(norm_content_feat.permute(0,2,1), norm_style_feat)

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

class AEAModule(nn.Module):
    def __init__(self, inplanes, scale_value=50, from_value=0.4, value_interval=0.5):
        super(AEAModule, self).__init__()
        self.inplanes = inplanes
        self.scale_value = scale_value
        self.from_value = from_value
        self.value_interval = value_interval

        self.f_psi = nn.Sequential(
            nn.Linear(self.inplanes, self.inplanes // 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.inplanes // 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, f_x):
        b, hw, c = x.size()
        clamp_value = self.f_psi(x.view(b * hw, c)) * self.value_interval + self.from_value
        clamp_value = clamp_value.view(b, hw, 1)
        clamp_fx = torch.sigmoid(self.scale_value * (f_x - clamp_value))
        return clamp_fx, clamp_value



class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1) # B * HW * C
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G) # B * HW * HW
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O
class AdaptiveSANet(nn.Module):
    def __init__(self, in_planes, spatial_dims):
        super(AdaptiveSANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.attention_layer = AEAModule(spatial_dims)
        self.claim_value = 0
        
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        affinity_matrix = cal_affinity_matrix(content,style) # B * HW * HW
        F = F.view(b, -1, w * h).permute(0, 2, 1) # B * HW * C
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G) # B * HW * HW
        S, claim_value = self.attention_layer(affinity_matrix, S) # B * HW * HW
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        self.claim_value = claim_value
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes = in_planes)
        self.sanet5_1 = SANet(in_planes = in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))

class AdaptiveTransform(nn.Module):
    def __init__(self, in_planes, relu4_1_dims, relu5_1_dims):
        super(AdaptiveTransform, self).__init__()
        self.sanet4_1 = AdaptiveSANet(in_planes = in_planes,spatial_dims= relu4_1_dims)
        self.sanet5_1 = AdaptiveSANet(in_planes = in_planes, spatial_dims= relu5_1_dims)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))

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



class SAModel(nn.Module):
    def __init__(self, config,encoder, start_iter, img_size):
        super(SAModel, self).__init__()
        self.config = config
        encoder = nn.Sequential(*list(encoder.children())[:44])
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        self.transform = Transform(in_planes = 512)
        self.decoder = decoder
        if(start_iter > 0):
            self.transform.load_state_dict(torch.load('transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
               
    def test(self, content, style, iterations=0):
        self.eval()
        with torch.no_grad():
            style_feats = self.encode_with_intermediate(style)
            content_feats = self.encode_with_intermediate(content)
            fusion = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
            stylized = self.decoder(fusion)
            self.train()
            return stylized
    
    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm = True) + self.calc_content_loss(g_t_feats[4], content_feats[4], norm = True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        """IDENTITY LOSSES"""
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])
        
        total_loss = self.config['content_weight'] * loss_c + self.config['style_weight'] * loss_s + self.config['l_identity1_weight'] * l_identity1 + self.config['l_identity2_weight'] * l_identity2
        return {
            'style_loss': loss_s,
            'content_loss': loss_c,
            'l_identity1_loss': l_identity1,
            'l_identity2_loss': l_identity2,
            'total_loss': total_loss
        }, total_loss


class AdaptiveSAModel(nn.Module):
    def __init__(self, config,encoder, start_iter, img_size):
        super(AdaptiveSAModel, self).__init__()
        self.config = config
        encoder = nn.Sequential(*list(encoder.children())[:44])
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        #transform
        self.relu4_1_dims = (img_size // 2 ** 3) ** 2
        self.relu5_1_dims = (img_size // 2 ** 4) ** 2
        self.transform = AdaptiveTransform(in_planes = 512,relu4_1_dims=self.relu4_1_dims,relu5_1_dims= self.relu5_1_dims)
        self.decoder = decoder
        if(start_iter > 0):
            self.transform.load_state_dict(torch.load('transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
               
    def test(self, content, style, iterations=0):
        self.eval()
        with torch.no_grad():
            style_feats = self.encode_with_intermediate(style)
            content_feats = self.encode_with_intermediate(content)
            fusion = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
            stylized = self.decoder(fusion)
            relu4_1_claim_value = self.transform.sanet4_1.claim_value.detach().cpu().numpy()
            relu5_1_claim_value = self.transform.sanet5_1.claim_value.detach().cpu().numpy()
            for i, clam_value_map in enumerate(relu5_1_claim_value):
                seaborn.heatmap(data=clam_value_map, vmin=0, vmax=1)
                clamp_output = os.path.join(self.config['output'],'claim_map', f'{iterations}')
                if not os.path.exists(clamp_output):
                    os.makedirs(clamp_output, exist_ok=True)
                plt.show()
                plt.savefig(os.path.join(clamp_output, f'iteration_{iterations}_batch_{i}.png'))
            plt.clf()
            plt.close()
            self.train()
            return stylized
    
    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm = True) + self.calc_content_loss(g_t_feats[4], content_feats[4], norm = True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        """IDENTITY LOSSES"""
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])
        
        total_loss = self.config['content_weight'] * loss_c + self.config['style_weight'] * loss_s + self.config['l_identity1_weight'] * l_identity1 + self.config['l_identity2_weight'] * l_identity2
        return {
            'style_loss': loss_s,
            'content_loss': loss_c,
            'l_identity1_loss': l_identity1,
            'l_identity2_loss': l_identity2,
            'total_loss': total_loss
        }, total_loss
