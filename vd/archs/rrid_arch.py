import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=((kernel_size - 1) * dilation // 2), bias=bias, stride=stride, dilation=dilation)


## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Dilated Channel Attention Block (DCAB)
class DCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, dilation=1):
        super(DCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dilation=dilation))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dilation=dilation))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class GFM(nn.Module):
    def __init__(self, n_feat, bias=True, padding_mode='reflect'):
        super(GFM, self).__init__()
        self.pwconv = nn.Conv2d(n_feat, n_feat * 2, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(n_feat * 2, n_feat * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode,
                                groups=n_feat * 2)
        self.project_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.mlp = nn.Conv2d(n_feat, n_feat, 1, 1, 0, bias=True)

    def forward(self, x):
        shortcut = x
        x = self.pwconv(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return self.mlp(x + shortcut)


class Encoder_1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder_1, self).__init__()

        self.encoder_level1 = [DCAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [DCAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.encoder_level3 = [DCAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder_1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, pool='avg'):
        super(Decoder_1, self).__init__()

        self.decoder_level1 = [DCAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [DCAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.decoder_level3 = [DCAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        # self.skip_attn1 = DCAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        # self.skip_attn2 = DCAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
        self.skip_attn1 = GFM(n_feat)
        self.skip_attn2 = GFM(n_feat + scale_unetfeats)
        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


class Encoder_2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder_2, self).__init__()

        self.encoder_level1 = [DCAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [DCAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.encoder_level3 = [DCAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder_2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, pool='avg'):
        super(Decoder_2, self).__init__()

        self.decoder_level1 = [DCAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [DCAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act, dilation=2)
                               for _ in range(2)]
        self.decoder_level3 = [DCAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act,
                                    dilation=4) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = FSM(num_feat=n_feat, block_size=8)
        self.skip_attn2 = FSM(num_feat=n_feat + scale_unetfeats, block_size=8)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


class SCAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, cha, reduction):
        super(SCAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(n_feat, n_feat // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(n_feat // reduction, n_feat, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(
            nn.Conv2d(n_feat, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())

        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = nn.Sequential(conv(cha, n_feat, kernel_size, bias=bias), nn.PReLU())
        self.conv3 = nn.Sequential(conv(cha, n_feat, kernel_size, bias=bias), nn.PReLU())

    def forward(self, x, img):
        x1 = self.conv1(x)

        x2 = self.conv2(img)
        fea_c = self.avg_pool(x2)
        fea_c = self.fc1(fea_c)
        fea_c = self.relu(fea_c)
        fea_c = self.fc2(fea_c)
        chn_se = self.sigmoid(fea_c)
        chn_se = chn_se * x1

        x3 = self.conv3(img)
        spa_se = self.spatial_se(x3)
        out = chn_se * spa_se
        return out


# class RRT(nn.Module):
#     def __init__(self, n_feat, kernel_size, bias):
#         super(RRT, self).__init__()
#         self.num_heads = 4
#         self.temperature = nn.Parameter(torch.ones(4, 1, 1))
#         # self.upsample = nn.Sequential(conv(n_feat, n_feat * 4, kernel_size, bias=bias), nn.PReLU(),
#         #                           nn.PixelShuffle(2),
#         #                           conv(n_feat, n_feat, kernel_size, bias=bias))
#         self.conv = nn.Conv2d(n_feat, n_feat*3, kernel_size=1, bias=bias)
#         self.project_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
#         self.feedforward = nn.Sequential(
#             nn.Conv2d(n_feat, n_feat, 1, 1, 0, bias=bias),
#             nn.GELU(),
#             nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=bias, groups=n_feat, padding_mode='reflect'),
#             nn.GELU()
#         )
#
#     def forward(self, x):
#         # f = self.upsample(x)
#         qkv = self.conv(x)
#         _, _, h, w = qkv.shape
#         q, k, v = qkv.chunk(3, dim=1)
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         out = self.feedforward(0.1 * out + x)
#         return out

class RGISP(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RGISP, self).__init__()
        self.num_heads = 4
        self.temperature = nn.Parameter(torch.ones(4, 1, 1))
        # self.upsample = nn.Sequential(conv(n_feat, n_feat * 4, kernel_size, bias=bias), nn.PReLU(),
        #                           nn.PixelShuffle(2),
        #                           conv(n_feat, n_feat, kernel_size, bias=bias))

        # self.upsample = nn.Sequential(conv(n_feat, n_feat * 4, 3, bias=False), nn.PReLU(),
        #                               nn.PixelShuffle(2),
        #                               conv(n_feat, n_feat, 3, bias=False))
        self.conv_raw = nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, bias=bias)
        self.conv_rgb = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=bias, groups=n_feat, padding_mode='reflect'),
            nn.GELU()
        )

    def forward(self, rgb, raw):
        # raw_up = self.upsample(raw)
        qk = self.conv_raw(raw)
        v = self.conv_rgb(rgb)

        _, _, h, w = qk.shape
        q, k = qk.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.feedforward(out + raw)
        return out


class RB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(RB, self).__init__()
        modules_body = []
        modules_body = [DCAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResNet(nn.Module):
    def __init__(self, n_feat, scale_resnetfeats, kernel_size, reduction, act, bias, scale_edecoderfeats, num_cab):
        super(ResNet, self).__init__()

        self.orb1 = RB(n_feat + scale_resnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = RB(n_feat + scale_resnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = RB(n_feat + scale_resnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_edecoderfeats)
        self.up_dec1 = UpSample(n_feat, scale_edecoderfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_edecoderfeats, scale_edecoderfeats),
                                     UpSample(n_feat, scale_edecoderfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_edecoderfeats, scale_edecoderfeats),
                                     UpSample(n_feat, scale_edecoderfeats))

        self.conv_enc1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))
        self.conv_enc2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))
        self.conv_enc3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))

        self.conv_dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))
        self.conv_dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))
        self.conv_dec3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(n_feat, n_feat + scale_resnetfeats, kernel_size=1, bias=bias))

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


class dct_weight(nn.Module):
    def __init__(self, in_c=40, out_c=40):
        super(dct_weight, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class directional_dct_layer(nn.Module):
    def __init__(self, in_c=40, out_c=40):
        super(directional_dct_layer, self).__init__()
        self.ddct_conv = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.GELU()
        self.conv_dct = nn.Conv2d(in_c, out_c, kernel_size=3, stride=8, padding=2, dilation=2, groups=in_c)

    def forward(self, x):
        out = self.act(self.ddct_conv(x)) + x
        out = self.conv_dct(out)
        return out


def check_image_size(h, w, bs):
    new_h = h
    new_w = w
    if h % bs != 0:
        new_h = h + (bs - h % bs)
    if w % bs != 0:
        new_w = w + (bs - w % bs)
    return new_h, new_w


def IDDCTmode0(im):
    C, M, N = im.shape
    DD = torch.zeros((C, M, N))
    D = torch.zeros((C, M, N))

    for i in range(N):
        DD[:, :, i] = torch.fft.ifft(im[:, :, i], norm='ortho', dim=-1).real

    for j in range(M):
        D[:, j, :] = torch.fft.ifft(DD[:, j, :], norm='ortho', dim=-1).real
    return D


def directional_inverse_dct_layer(img, bs=8, mode=0):
    b, ch, h, w = img.shape
    imt = img.view(b * ch, h, w)
    c, m, n = imt.shape
    new_m, new_n = check_image_size(m, n, bs)
    new_imt = torch.zeros((c, new_m, new_n)).cuda()
    new_imt[:, :m, :n] = imt
    imf = torch.zeros((c, new_m, new_n)).cuda()
    for ii in range(0, new_m, bs):
        for jj in range(0, new_n, bs):
            cb = new_imt[:, ii:ii + bs, jj:jj + bs]
            # CB = DDCT_transform(cb, mode)
            # cbf = IDDCT(CB, mode)
            CB = cb
            cbf = IDDCTmode0(CB)
            imf[:, ii:ii + bs, jj:jj + bs] = cbf
    imf = imf[:, :m, :n]
    didct = imf.view(b, ch, h, w)
    return didct


class FSM(nn.Module):
    def __init__(self, num_feat=40, block_size=8):
        super(FSM, self).__init__()
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_dct_0 = directional_dct_layer(in_c=num_feat, out_c=num_feat)

        self.dct_weight = dct_weight(in_c=num_feat, out_c=num_feat)
        self.in_c = num_feat
        self.bs = block_size

        self.after_rdct = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        _, _, h, w = x.size()
        dct_feat = self.act(self.conv(x))
        # mode 0
        dct_feat_0 = self.conv_dct_0(dct_feat)
        out_0 = directional_inverse_dct_layer(dct_feat_0, bs=self.bs, mode=0)
        out_0 = F.interpolate(out_0, size=(h, w), mode='bilinear', align_corners=False)

        dct_weight = self.dct_weight(x)
        out = torch.mul(out_0, dct_weight)

        out = self.after_rdct(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, n_feat=40):
        super(FusionBlock, self).__init__()
        self.upsample = nn.Sequential(conv(n_feat, n_feat * 4, 3, bias=False), nn.PReLU(),
                                      nn.PixelShuffle(2),
                                      conv(n_feat, n_feat, 3, bias=False))
        self.conv = nn.Conv2d(n_feat, n_feat, 1)
        self.concat2_r = conv(n_feat * 2, n_feat, 3, bias=False)

    def forward(self, rgb, raw):
        x1 = rgb
        x2 = self.upsample(raw)
        w1 = torch.sigmoid(self.conv(x1))
        w2 = torch.sigmoid(self.conv(x2))
        x1_ = torch.mul(x1, w1)
        x2_ = torch.mul(x2, w2)
        max = torch.maximum(x1_, x2_)
        avg = (x1_ + x2_) / 2
        out = self.concat2_r(torch.cat([max, avg], 1))
        out = F.relu(out)
        return out


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape  # torch.Size([1, 64, 64, 1])
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        # calculate flops for 1 window with token length of n
        flops = 0
        # qkv = self.qkv(x)
        flops += n * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        x_permute = x.permute(0, 3, 1, 2)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nw*b, window_size*window_size, c
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


# RAW-RGB-Image-Demoireing Model
@ARCH_REGISTRY.register()
class RRID(nn.Module):

    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_edecoderfeats=20, scale_resnetfeats=16, num_cab=8,
                 kernel_size=3, reduction=4, bias=False,
                 nf=40, depths=[4, 4, 4, 4], num_heads=[4, 4, 4, 4], window_size=8, mlp_ratio=2, qkv_bias=True,
                 norm_layer=nn.LayerNorm, img_size=64, patch_size=1, resi_connection='1conv'
                 ):
        super(RRID, self).__init__()
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(4, n_feat, kernel_size, bias=bias),
                                           DCAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(nn.Conv2d(in_c, n_feat, 2, 2),
                                           DCAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        self.stage1_encoder = Encoder_1(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats, csff=False)
        self.stage1_decoder = Decoder_1(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats, pool='avg')
        self.stage1_rconv = conv(n_feat, 4, kernel_size, bias=bias)
        # self.scam1 = SCAM(n_feat, kernel_size=1, bias=bias, cha=4, reduction=4)
        self.RGISP = RGISP(n_feat, kernel_size, bias)
        self.tail_r = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.stage2_encoder = Encoder_2(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats, csff=False)
        self.stage2_decoder = Decoder_2(n_feat, kernel_size, reduction, act, bias, scale_edecoderfeats, pool='avg')
        self.stage2_rconv = conv(n_feat, n_feat, kernel_size, bias=bias)
        # self.stage2_up = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4 ** 2, 3, 1, 1), nn.PixelShuffle(4))
        self.conv_first = nn.Conv2d(n_feat, nf, 2, 2)
        self.upsample = nn.Sequential(nn.Conv2d(nf, out_c * 4 ** 2, 3, 1, 1), nn.PixelShuffle(4))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=nf,
            embed_dim=nf,
            norm_layer=norm_layer)
        patches_resolution = self.patch_embed.patches_resolution

        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RSTB(
                dim=nf,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(nf)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=nf,
            embed_dim=nf,
            norm_layer=norm_layer)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward_feature(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, lq_rgb, lq_raw):
        ##-------------- Branch 1: RAW---------------------
        fea_raw = self.shallow_feat1(lq_raw)
        encoder_raw = self.stage1_encoder(fea_raw)
        decoder_raw = self.stage1_decoder(encoder_raw)

        out_raw = self.stage1_rconv(decoder_raw[0]) + lq_raw
        namfeats_raw = decoder_raw[0]

        ##-------------- Branch 2: RGB---------------------
        fea_rgb = self.shallow_feat2(lq_rgb)
        encoder_rgb = self.stage2_encoder(fea_rgb)
        decoder_rgb = self.stage2_decoder(encoder_rgb)
        c_rgb = self.stage2_rconv(decoder_rgb[0]) + fea_rgb
        # c_rgb = self.stage2_up(c_rgb)

        ##-------------- Fusion---------------------
        fea_isp = self.RGISP(c_rgb, namfeats_raw)
        out_rgb = self.conv_first(F.relu(self.tail_r(fea_isp)))
        out_rgb = self.conv_after_body(self.forward_feature(out_rgb)) + out_rgb
        out_rgb = self.upsample(out_rgb)
        return out_rgb, out_raw


if __name__ == '__main__':
    x = torch.rand(1, 3, 1088, 1920)
    model = RRID()
    print(model(x).shape)
