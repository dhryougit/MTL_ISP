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
from basicsr.utils.flops_util import count_model_param_flops, print_model_param_nums

from basicsr.models.archs.quant_ops  import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import math
import matplotlib.pyplot as plt

class Adaptive_freqfilter_regression(nn.Module):
    def __init__(self):
        super().__init__()

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        # self.down1 = nn.Conv2d(16, 32, 2, 2, bias=True)
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
                               
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1,bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1,bias=True)
        # self.down2 = nn.Conv2d(32, 64, 2, 2, bias=True)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=0)
        # reg4 setting
        self.radius_factor_set = torch.arange(0.01, 1.01, 0.01).cuda()
  
        self.fclayer_v1 = nn.Linear(64, 256)
        self.fclayer_v2 = nn.Linear(256, len(self.radius_factor_set))
        self.leaky_relu = nn.LeakyReLU()

   
    def forward(self, x):
        B, C, H, W = x.size()
        inp = x

        a, b = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = torch.sqrt((a - H/2)**2 + (b - W/2)**2)
        dist = dist.to(x.device)
        max_radius = math.sqrt(H*H+W*W)/2

       
        x = torch.fft.fftn(x, dim=(-1,-2))
        x = torch.fft.fftshift(x)

        x_mag = torch.abs(x)
        x_mag = torch.log10(x_mag + 1)

        filter_input = torch.cat((inp,x_mag), dim=1)
        y = self.conv1(filter_input)
        y = self.relu(y)
        y = self.down1(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.down2(y)
        y = self.conv3(y)


        y = self.avgpool(y)
        y = y.squeeze(-1)
        y = y.squeeze(-1)

        # print(y.size())

        # radius_factor_set = self.sig(self.fclayer_r2(self.fclayer_r1(y)))
        value_set =  self.leaky_relu(self.fclayer_v2(self.fclayer_v1(y)))
        # value_set =  self.sig(self.fclayer_v2(self.fclayer_v1(y)))
        radius_set = max_radius*self.radius_factor_set


        mask = []

        zero_mask = torch.zeros_like(dist).cuda()
        one_mask = torch.ones_like(dist).cuda()
        for i in range(len(radius_set)):
            if i == 0:
                mask.append(torch.where((dist <= radius_set[i]), one_mask, zero_mask))
            else :
                mask.append(torch.where((dist <= radius_set[i]) & (dist > radius_set[i-1]), one_mask, zero_mask))


        fq_mask_set = torch.stack(mask, dim=0)
      
        # value_set [B, 100, 1, 1], fq_mask_set [1, 100, H, W]
        fq_mask = value_set.unsqueeze(-1).unsqueeze(-1) * fq_mask_set.unsqueeze(0)
        fq_mask = torch.sum(fq_mask, dim=1)
        


        lowpass = (x*fq_mask.unsqueeze(1))

        lowpass = torch.fft.ifftshift(lowpass)

        lowpass = torch.fft.ifftn(lowpass, dim=(-1,-2))

        # lowpass = torch.abs(lowpass)
        lowpass = torch.clamp(lowpass.real, 0 , 1)


        return lowpass, fq_mask, value_set

class Adaptive_freqfilter_classification(nn.Module):
    def __init__(self):
        super().__init__()

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.down1 = nn.Conv2d(16, 32, 2, 2, bias=True)
                               
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1,bias=True)
        self.down2 = nn.Conv2d(32, 64, 2, 2, bias=True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.down3 = nn.Conv2d(64, 128, 2, 2, bias=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.temp = torch.tensor(1)
        

        self.fclayer_v1 = nn.Linear(128, 256)
        self.fclayer_v2 = nn.Linear(256, 5)

        # self.multset = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6 ,1.8, 2.0]).cuda()
        # self.multset = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ,0.9, 1.0]).cuda()
        # self.radius_factor_set = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ,0.9, 1.0]).cuda()

        self.radius_factor_set = torch.tensor([0.3, 0.5, 0.7, 1.0]).cuda()
        self.value_num = 4


    def forward(self, x):
        B, C, H, W = x.size()
        inp = x

        a, b = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = torch.sqrt((a - H/2)**2 + (b - W/2)**2)
        # dist = dist.repeat(B, 1, 1).to(x.device)
        dist = dist.to(x.device)
        max_radius = math.sqrt(H*H+W*W)/2

       
        x = torch.fft.fftn(x, dim=(-1,-2))
        x = torch.fft.fftshift(x)
        x_mag = torch.abs(x)
        x_mag = torch.log10(x_mag + 1)

        x_mag_max = torch.max(x_mag)
        x_fq = x_mag / x_mag_max
        y = self.conv1(x_mag)
        y = self.relu(y)
        y = self.down1(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.down2(y)
        y = self.conv3(y)
        y = self.relu(y)
        y = self.down3(y)
    
        y = self.avgpool(y)
        y = y.squeeze(-1)
        y = y.squeeze(-1)

        # print(y.size())

        # radius_factor_set = self.sig(self.fclayer_r2(self.fclayer_r1(y)))
        value_set =  self.sig(self.fclayer_v2(self.fclayer_v1(y)))
        # value_set =  self.relu(self.fclayer_v2(self.fclayer_v1(y)))
        # value_set =  self.fclayer_v2(self.fclayer_v1(y))
        # value_set = torch.mean(value_set, dim=0)
        # print(value_set)
        value_set = value_set / self.temp
        value_set = self.soft(value_set)
        value_set = value_set[:, :self.value_num]
        # print(value_set)
        # value_set_hw = value_set
        
        radius_set = max_radius*self.radius_factor_set

        # mask = [torch.sigmoid(radius_set[0].to(x.device) - dist.to(x.device)) * value_set_sum[0]]
        mask = []
        for i in range(self.value_num):
            mask.append((torch.sigmoid(radius_set[i].to(x.device) - dist.to(x.device))))
        
        # print(mask.shape)
        fq_mask_set = torch.stack(mask, dim=0)
        # fq_mask = torch.mul(value_set.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W), fq_mask_set.unsqueeze(0).expand(B, -1, -1, -1))
        fq_mask = value_set.unsqueeze(-1).unsqueeze(-1) * fq_mask_set.unsqueeze(0)
   
        fq_mask = torch.sum(fq_mask, dim=1)

        # x = torch.fft.fftn(x, dim=(-1,-2))
        # x = torch.fft.fftshift(x)
        # lowpass = (x*fq_mask)
        lowpass = (x*fq_mask.unsqueeze(1))

        lowpass = torch.fft.ifftshift(lowpass)

        lowpass = torch.fft.ifftn(lowpass, dim=(-1,-2))

        lowpass = torch.abs(lowpass)
        # lowpass = lowpass.real
        
        return lowpass,  value_set


class Lowpassfilter(nn.Module):
    def __init__(self):
        super().__init__()

        self.radius1 = nn.Parameter(torch.tensor(0.3))
        self.radius1_val = nn.Parameter(torch.tensor(1.0))

        self.radius2 = nn.Parameter(torch.tensor(0.5))
        self.radius2_val = nn.Parameter(torch.tensor(0.7))

        self.radius3 = nn.Parameter(torch.tensor(0.7))
        self.radius3_val = nn.Parameter(torch.tensor(0.5))

        self.radius4 = nn.Parameter(torch.tensor(1.0))
        self.radius4_val = nn.Parameter(torch.tensor(0.2))

     


    def forward(self, x):
        B, C, H, W = x.size()
        inp = x

        a, b = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = torch.sqrt((a - H/2)**2 + (b - W/2)**2)
        
        # radius = math.sqrt(H*H+W*W)/self.alpha
        radius1 = (math.sqrt(H*H+W*W)/2)*self.radius1
        radius2 = (math.sqrt(H*H+W*W)/2)*self.radius2
        radius3 = (math.sqrt(H*H+W*W)/2)*self.radius3
        radius4 = (math.sqrt(H*H+W*W)/2)*self.radius4
   
        # mask = dist < radius.to(dist.device)
        mask1 = torch.sigmoid(radius1.to(x.device) - dist.to(x.device)) * self.radius1_val
        mask2 = (torch.sigmoid(radius2.to(x.device) - dist.to(x.device)) - torch.sigmoid(radius1.to(x.device) - dist.to(x.device))) * self.radius2_val
        mask3 = (torch.sigmoid(radius3.to(x.device) - dist.to(x.device)) - torch.sigmoid(radius2.to(x.device) - dist.to(x.device))) * self.radius3_val
        mask4 = (torch.sigmoid(radius4.to(x.device) - dist.to(x.device)) - torch.sigmoid(radius3.to(x.device) - dist.to(x.device))) * self.radius4_val
   
        # mask = torch.clamp(mask1+mask2+mask3+mask4+mask5+mask6+mask7+mask8, 0, 1)
        mask = mask1+mask2+mask3+mask4


        lpf = mask.to(torch.float32).to(x.device)
   
        x = torch.fft.fftn(x, dim=(-1,-2))
        x = torch.fft.fftshift(x)

        lowpass = (x*lpf)

        lowpass = torch.fft.ifftshift(lowpass)

        lowpass = torch.fft.ifftn(lowpass, dim=(-1,-2))

        lowpass = torch.abs(lowpass)

        return lowpass
class Fix_lowpassfilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.tensor(0.7)


    def forward(self, x):
        B, C, H, W = x.size()
        inp = x

        a, b = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = torch.sqrt((a - H/2)**2 + (b - W/2)**2)
        
        # print(self.alpha)
        radius = (math.sqrt(H*H+W*W)/2)*self.factor
        
        # mask = dist < radius.to(dist.device)
        mask = torch.sigmoid(radius.to(dist.device) - dist)
        lpf = mask.to(torch.float32).to(x.device)
   
        x = torch.fft.fftn(x, dim=(-1,-2))
        x = torch.fft.fftshift(x)

        lowpass = (x*lpf)

        lowpass = torch.fft.ifftshift(lowpass)

        lowpass = torch.fft.ifftn(lowpass, dim=(-1,-2))

        lowpass = torch.abs(lowpass)

        return lowpass, mask

        
        
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.norm2 = LayerNorm2d(c)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
        

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        
        

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        # print(x.grad)

        return y + x * self.gamma




class NAFNet_filter(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # self.filter = Adaptive_freqfilter_classification()
        # self.filter = Lowpassfilter()
        self.filter = Adaptive_freqfilter_regression()
        # self.filter = Lowpassfilter()
        self.mask = {}

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
        # x = self.intro(inp)

        # if mode == 'on':
        #     x = self.filter(inp)[0]
        #     filtered = torch.cat((inp,x), dim=1)
        #     x = self.intro(filtered)
        # else : 
        #     dummy = torch.zeros_like(inp).cuda()
        #     x = torch.cat((inp,dummy), dim=1)
        #     x = self.intro(x)


        x = self.filter(inp)[0]
        x = self.intro(x)

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

class NAFNetLocal(Local_Base, NAFNet_filter):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet_filter.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    # width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    # width = 64
    # enc_blks =  [2, 2, 4, 8]
    # middle_blk_num =  0
    # dec_blks =  [2, 2, 2, 2]

    width = 16
    enc_blks = [2, 2, 2, 2]
    middle_blk_num = 2
    dec_blks = [2, 2, 2, 2]

    # width = 32
    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    # width = 64
    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    net = NAFNet_filter(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

  
    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info
    from torchsummary import summary as summary_

    summary_(net.cuda(),(3, 256, 256),batch_size=1)

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)

    # net = net.cpu()
    flops = count_model_param_flops(net)
    params = print_model_param_nums(net)
    print(flops, params)
