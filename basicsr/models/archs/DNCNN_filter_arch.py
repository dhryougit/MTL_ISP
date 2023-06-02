import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from models import register_model
# from models import BFBatchNorm2d
import math

class Random_frequency_replacing(nn.Module):
    def __init__(self):
        super().__init__()

   
        self.radius_factor_set = torch.arange(0.01, 1.01, 0.01).cuda()
        self.value_set = torch.arange(1., 0., -0.01).cuda()

   
    def forward(self, clean, noisy):
        B, C, H, W = noisy.size()
        inp = noisy

        a, b = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = torch.sqrt((a - H/2)**2 + (b - W/2)**2)
        dist = dist.to(noisy.device)
        max_radius = math.sqrt(H*H+W*W)/2

       
        noisy_fq = torch.fft.fftn(noisy, dim=(-1,-2))
        noisy_fq = torch.fft.fftshift(noisy_fq)
        clean_fq = torch.fft.fftn(clean, dim=(-1,-2))
        clean_fq = torch.fft.fftshift(clean_fq)


       
        radius_set = max_radius*self.radius_factor_set

        value_prob = self.value_set.view(1, -1)
        value_prob = value_prob.repeat(B, 1)

        value_set =  torch.bernoulli(value_prob*0.3).cuda()
        # value_set =  torch.bernoulli(torch.ones(B,len(self.radius_factor_set))*0.2).cuda()
       

        mask = []
        # zero = torch.tensor(0.0, dtype=torch.float32).cuda()
        zero_mask = torch.zeros_like(dist).cuda()
        one_mask = torch.ones_like(dist).cuda()
        for i in range(len(radius_set)):
            if i == 0:
                mask.append(torch.where((dist < radius_set[i]), one_mask, zero_mask))
            else :
                mask.append(torch.where((dist < radius_set[i]) & (dist >= radius_set[i-1]), one_mask, zero_mask))
           

        fq_mask_set = torch.stack(mask, dim=0)
        fq_mask = value_set.unsqueeze(-1).unsqueeze(-1) * fq_mask_set.unsqueeze(0)
        fq_mask = torch.sum(fq_mask, dim=1)
        # print(fq_mask[0])
    
        # fq_mask [B,H,W]
        # bn1이 작은 확률 --> noise에 넣어줌 
        bn1_mask = fq_mask
        bn2_mask = torch.ones_like(bn1_mask)-bn1_mask


        noisy_fq = (noisy_fq*bn1_mask.unsqueeze(1))
        clean_fq = (clean_fq*bn2_mask.unsqueeze(1))

        replaced_fq = noisy_fq+clean_fq
        replaced_fq = torch.fft.ifftshift(replaced_fq)

        replaced_fq = torch.fft.ifftn(replaced_fq, dim=(-1,-2))

        replaced_fq = replaced_fq.real
        # lowpass = torch.clamp(lowpass.real, 0 , 1)


        return replaced_fq


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



class Highpassfilter(nn.Module):
    def __init__(self):
        super().__init__()

        self.radius1 = nn.Parameter(torch.tensor(0.3))
        self.radius1_val = nn.Parameter(torch.tensor(1.0))

        self.radius2 = nn.Parameter(torch.tensor(0.5))
        self.radius2_val = nn.Parameter(torch.tensor(1.0))

        self.radius3 = nn.Parameter(torch.tensor(0.7))
        self.radius3_val = nn.Parameter(torch.tensor(1.0))

        self.radius4 = nn.Parameter(torch.tensor(1.0))
        self.radius4_val = nn.Parameter(torch.tensor(0.0))

     


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
        



# @register_model("dncnn")
class DNCNN_filter(nn.Module):
    """DnCNN as defined in https://arxiv.org/abs/1608.03981 
        reference implementation: https://github.com/SaoYan/DnCNN-PyTorch"""
    def __init__(self, depth=20, n_channels=64, image_channels=3, bias=False, kernel_size=3):
        super(DNCNN_filter, self).__init__()
        kernel_size = 3
        padding = 1

        self.bias = bias
        # self.filter = Adaptive_freqfilter_regression()
        # self.filter = Highpassfilter()
        # if not bias:
        # 	norm_layer = BFBatchNorm2d.BFBatchNorm2d
        # else:

        # norm_layer = nn.BatchNorm2d

        self.depth = depth
        self.replace = Random_frequency_replacing()


        self.first_layer = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)

        self.hidden_layer_list = [None] * (self.depth - 2)

        # self.bn_layer_list = [None] * (self.depth -2 );

        for i in range(self.depth-2):
            self.hidden_layer_list[i] = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)
            # self.bn_layer_list[i] = norm_layer(n_channels)

        self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list)
        # self.bn_layer_list = nn.ModuleList(self.bn_layer_list);
        self.last_layer = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)

        self.feautre_to_img1 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)
        # self.feautre_to_img2 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)
        # self.feautre_to_img3 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)

        self._initialize_weights()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--in-channels", type=int, default=1, help="number of channels")
        parser.add_argument("--hidden-size", type=int, default=64, help="hidden dimension")
        parser.add_argument("--num-layers", default=20, type=int, help="number of layers")
        parser.add_argument("--bias", action='store_true', help="use residual bias")

    @classmethod
    def build_model(cls, args):
        return cls(image_channels = args.in_channels, n_channels = args.hidden_size, depth = args.num_layers, bias=args.bias)

    def forward(self, x, swap='false', feature=None):
        y = x
        # x = self.filter(x)[0]
        # x = self.filter(x)
        # filtered = torch.cat((y,x), dim=1)
        out = self.first_layer(x)
        out = F.relu(out)

        for i in range(self.depth-2):
            out = self.hidden_layer_list[i](out)
            # out = self.bn_layer_list[i](out);
            out = F.relu(out)
            if i == 8 :
                img = self.feautre_to_img1(out)
                if swap =='true':
                    out = self.replace(out,feature)

            # elif i == 8 :
            #     img = self.feautre_to_img2(out)
            # elif i == 13 :
            #     img = self.feautre_to_img3(out)

        out = self.last_layer(out)
        
        return torch.clamp(y-out, 0, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d) or isinstance(m, BFBatchNorm2d.BFBatchNorm2d):
            #     m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
            #     init.constant_(m.bias, 0)

if __name__ == '__main__':
    img_channel = 3
    # width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    width = 64
    enc_blks =  [2, 2, 4, 8]
    middle_blk_num =  12
    dec_blks =  [2, 2, 2, 2]

    # width = 64
    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    net = DNCNN_filter()

  
    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info
    from torchsummary import summary as summary_

    # summary_(net.cuda(),(3, 256, 256),batch_size=1)

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
