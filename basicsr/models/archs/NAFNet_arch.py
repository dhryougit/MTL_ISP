# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        ffted = torch.fft.rfftn(x,s=(h,w),dim=(2,3),norm='ortho')
        ffted = torch.cat([ffted.real,ffted.imag],dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = torch.tensor_split(ffted,2,dim=1)
        ffted = torch.complex(ffted[0],ffted[1])
        output = torch.fft.irfftn(ffted,s=(h,w),dim=(2,3),norm='ortho')

        # (batch, c, h, w/2+1, 2)
        # ffted = torch.rfft(x, signal_ndim=2, normalized=True)
        # print(x.size())
        # ffted = torch.fft.rfft2(x)
        # print(ffted.size())

        # ffted_shifted = torch.empty(batch, 2*c, h, int(w/2+1)).cuda()
        # for i in range(c):
        #     ffted_shifted[:,i] = ffted[:,i].real
        #     ffted_shifted[:,i+c] = ffted[:,i].imag

        # print(ffted_shifted.size())

        
        # (batch, c, 2, h, w/2+1)
        # ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        # ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # ffted = self.conv_layer(ffted_shifted)  # (batch, c*2, h, w/2+1)
        # ffted = self.relu(self.bn(ffted))

        # ffted_shifted_2 = torch.empty((batch, c, h, int(w/2+1)), dtype=torch.cfloat).cuda()
        # for i in range(c):
        #     ffted_shifted_2[:,i].real = ffted[:,i]
        #     ffted_shifted_2[:,i].imag = ffted[:,i+c]

        # ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
        #     0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        # output = torch.irfft(ffted, signal_ndim=2, signal_sizes=r_size[2:], normalized=True)
        # output = torch.fft.irfft2(ffted, x.numel() )
        # output = torch.fft.irfft2(ffted_shifted_2, x.numel())
        # torch.testing.assert_close(output, x, check_stride=False)

        # output = torch.fft.irfft2(ffted_shifted_2)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        ratio = 0.5
        self.ratio = ratio
        c_split = int(c*ratio)
        dw_split = int(dw_channel*ratio)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
        #                        bias=True)

        # self.conv_l2l_1 = nn.Conv2d(in_channels=c_split, out_channels=dw_split, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_l2l_2 = nn.Conv2d(in_channels=dw_split, out_channels=dw_split, kernel_size=3, padding=1, stride=1, groups=dw_split,
                               bias=True)
        # self.conv_l2g_1 = nn.Conv2d(in_channels=c_split, out_channels=dw_split, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_l2g_2 = nn.Conv2d(in_channels=dw_split, out_channels=dw_split, kernel_size=3, padding=1, stride=1, groups=dw_split,
                               bias=True)

        # self.conv_g2l_1 = nn.Conv2d(in_channels=c_split, out_channels=dw_split, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_g2l_2 = nn.Conv2d(in_channels=dw_split, out_channels=dw_split, kernel_size=3, padding=1, stride=1, groups=dw_split,
                               bias=True)
        self.conv_g2g = SpectralTransform(dw_split, dw_split, stride=1)


        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        ffn_split = int(ffn_channel*ratio)
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_l2l_4 = nn.Conv2d(in_channels=c_split, out_channels=ffn_split, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_l2g_4 = nn.Conv2d(in_channels=c_split, out_channels=ffn_split, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_g2l_4 = nn.Conv2d(in_channels=c_split, out_channels=ffn_split, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_g2g_4 = SpectralTransform(c_split, ffn_split, stride=1)

        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
   
        x = self.norm1(x)
        x = self.conv1(x)
        # x = self.conv2(x)
        x_size = int(x.size(1)*self.ratio)
        x_l = x[:,:x_size]
        x_g = x[:,x_size:]

        # out_xl_1 = self.conv_l2l_1(x_l)
        out_xl_1 = self.conv_l2l_2(x_l)

        # out_xl_2 = self.conv_g2l_1(x_g)
        out_xl_2 = self.conv_g2l_2(x_g)

        # out_xg_1 = self.conv_l2g_1(x_l)
        out_xg_1 = self.conv_l2g_2(x_l)

        out_xg_2 = self.conv_g2g(x_g)

        out_xl = out_xl_1 + out_xl_2
        out_xg = out_xg_1 + out_xg_2

        # x = self.sg(x)
        x = out_xl*out_xg

        x = x * self.sca(x)
        x = self.conv3(x)
        

        x = self.dropout1(x)

        y = inp + x * self.beta

        # x = self.conv4(self.norm2(y))
        x = self.norm2(y)
        x_size = int(x.size(1)*self.ratio)

        x_l = x[:,:x_size]
        x_g = x[:,x_size:]

        out_xl_1 = self.conv_l2l_4(x_l)

        out_xl_2 = self.conv_g2l_4(x_g)

        out_xg_1 = self.conv_l2g_4(x_l)

        out_xg_2 = self.conv_g2g_4(x_g)


        out_xl = out_xl_1 + out_xl_2
        out_xg = out_xg_1 + out_xg_2

        # x = self.sg(x)
        x = out_xl*out_xg

        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    # width = 32

    # # enc_blks = [2, 2, 4, 8]
    # # middle_blk_num = 12
    # # dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]

    width = 32
    enc_blks =  [2, 2, 4, 8]
    middle_blk_num =  12
    dec_blks =  [2, 2, 2, 2]

    # width = 64
    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).cuda()


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
