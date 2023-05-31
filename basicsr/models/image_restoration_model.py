# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import time
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.mask import Masker

import matplotlib.pyplot as plt

import wandb
import sys

from advertorch.attacks4IP.zero_mean_pgd import L2PGDAttack
import math
torch.autograd.set_detect_anomaly(True)


loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

def forward_hook(module, input, output):
    module.feature_map = output
    inputs = input[0]
    module.input = inputs


class Random_frequency_replacing(nn.Module):
    def __init__(self, fbr_param=0.5, mode='linear'):
        super().__init__()

   
        self.radius_factor_set = torch.arange(0.01, 1.01, 0.01).cuda()
        if mode == 'linear':
            print('linear')
            self.value_set = torch.arange(1., 0., -0.01).cuda()
        else:
            self.value_set = torch.exp(torch.linspace(0, -10, 100)).cuda()
        self.fbr_param = fbr_param
        

   
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

        value_set =  torch.bernoulli(value_prob*self.fbr_param).cuda()
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


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.prune_rate = 0
        self.test_mode = 'ori'
        self.Random_frequency_replacing = Random_frequency_replacing(fbr_param=self.opt['train']['fbr_param'], mode=self.opt['train']['fbr_mode'])
        self.filter_on = 'on'
        self.masker = Masker(width = 3, mode='zero')
        self.mseloss =  nn.MSELoss()
        

        self.alpha = self.opt['train']['alpha']
        eps = 5
        patch_size=50
        l2_adv_tr = eps*1./255 * math.sqrt(patch_size ** 2)
        attack = (L2PGDAttack, dict(loss_fn=nn.MSELoss(), 
                        eps=l2_adv_tr, nb_iter=1, eps_iter=1*l2_adv_tr, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False))
        self.adversary = attack[0](self.net_g, **attack[1])
        
        

        self.maxvalue = []
   


      
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        

        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])


        for name, module in self.net_g.named_modules():
            if name == 'module.feautre_to_img1':
                module.register_forward_hook(forward_hook)

        for name, module in self.net_g.named_modules():
            if name == 'module.filter' :
                module.register_forward_hook(forward_hook)

        # module.hidden_layer_list.3

   


    def get_radius_set(self):
        radius_set = {}

        for k, v in self.net_g.named_parameters():
            if 'filter' in k :
                radius_set[k] = v.data.item()

        return  radius_set


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_filter = []

        for k, v in self.net_g.named_parameters():
            if 'filter' in k:
                if v.requires_grad:
                    optim_params_filter.append(v)
            else : 
                if v.requires_grad:
                    optim_params.append(v)

        # for k, v in self.net_g.named_parameters():
        #     if v.requires_grad:
        #         optim_params.append(v)
  
   

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
            self.optimizer_g_filter = torch.optim.Adam([{'params': optim_params_filter}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
            self.optimizer_g_filter = torch.optim.SGD([{'params': optim_params_filter}],
                                                **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            self.optimizer_g_filter = torch.optim.AdamW([{'params': optim_params_filter}],
                                                **train_opt['optim_g'])
            
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_g_filter)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq


    def gaussian_fgsm(self, model, x, y, epsilon=0.1, alpha=0.03, num_iter=1):
    # def pgd_attack(self, model, x, y, epsilon=0.1, alpha=0.01, num_iter=40):
        # Create a copy of the input tensor
        x_perturbed = x.clone().detach()

        # PGD attack loop
        for t in range(num_iter):
            # Forward pass to compute the model's output and loss
            x_perturbed.requires_grad = True
            
            output = model(x_perturbed)

            loss = self.cri_pix(output, y)
 
            # Backward pass to compute the gradient of the loss w.r.t. the input
            model.zero_grad()

            loss.backward()
            grad = x_perturbed.grad 
            # B,C,H,W = self.lq.size()
            # random_noise = torch.randn(B,C,H,W).cuda() * 0.2

            B,C,H,W = self.lq.size()
            random_noise = torch.randn(B,C,H,W).cuda()
            adv_random_noise = torch.abs(random_noise*0.2) * torch.sign(grad)

   

            adv_random_noise = adv_random_noise.cpu().numpy()
            adv_random_noise = adv_random_noise.reshape(adv_random_noise.shape[0], -1)

            # plt.hist(grad.flatten(), bins=500, range = [-0.1, 1.1])
            

            random_noise_dist = random_noise.cpu().numpy()*0.2
            random_noise_dist = random_noise_dist.reshape(random_noise_dist.shape[0], -1)
    


            # exit()
            
            # Add perturbation to the input
            with torch.no_grad():
                B,C,H,W = self.lq.size()
                random_noise = torch.randn(B,C,H,W).cuda()
                x_perturbed = x_perturbed + torch.abs(random_noise*0.2) * torch.sign(grad)
                # x_perturbed = torch.min(torch.max(x_perturbed, x - epsilon), x + epsilon)
                x_perturbed = torch.clamp(x_perturbed, 0, 1)
            
        
        return x_perturbed.detach()


    def pgd_attack(self, model, x, y, epsilon=0.3, alpha=(16/255), num_iter=1):
    # def pgd_attack(self, model, x, y, epsilon=0.1, alpha=0.01, num_iter=40):
        B,C,H,W = x.size()
        x_pgd = x.clone().detach()
        alpha=(self.opt['train']['perturb']['alpha']/255)
        num_iter = self.opt['train']['perturb']['iter']
        # PGD attack loop
        for t in range(num_iter):
            # Forward pass to compute the model's output and loss
            x_pgd.requires_grad = True
            
            output = model(x_pgd)

            loss = self.cri_pix(output, y) + 0. * sum(p.sum() for p in model.parameters())
 
            # Backward pass to compute the gradient of the loss w.r.t. the input
            model.zero_grad()

            loss.backward()
            # grad = torch.clamp(x_pgd.grad, -(25/255), 25/255)
            grad = x_pgd.grad

        
            # alpha = (torch.rand(B) * 16) * 1./255
            # alpha = alpha.cuda()
         
    
            
            # Add perturbation to the input
            with torch.no_grad():
                # pgd_grad_mean = torch.sign(grad).mean(dim=(2,3), keepdim=True)
                # pgd_grad = torch.sign(grad)-pgd_grad_mean

                # x_gpd = x_pgd + torch.sign(grad) * alpha.view(-1,1,1,1)
                x_pgd = x_pgd + alpha * torch.sign(grad)
                x_pgd = torch.min(torch.max(x_pgd, x - epsilon), x + epsilon)
                x_pgd = torch.clamp(x_pgd, 0, 1)
                # x_pgd = torch.clamp(x_pgd + grad, 0, 1)
            
        
        return x_pgd.detach()


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def obsattack(self):
        self.net_g.eval()
        self.set_requires_grad([self.net_g], False)

        noise = self.lq
        data_adv = self.adversary.perturb(noise, self.gt)

        self.set_requires_grad([self.net_g], True)
        self.net_g.train()

        return data_adv, noise
        
    def genearte_poisson_noise(self):
        B,C,H,W = self.lq.size()
        sigma = (torch.rand(B) * 55) * 1./255
        sigma = sigma.cuda()
        noise = torch.from_numpy(np.random.poisson(lam=1, size=(B,C,H, W))).float().cuda()
        noise = (noise - noise.mean()) / noise.std() * sigma.view(-1,1,1,1)
        return noise

    def genearte_gaussian_noise(self):
        B,C,H,W = self.lq.size()
        sigma = (torch.rand(B) * 55) * 1./255
        sigma = sigma.cuda()
        random_noise = torch.randn(B,C,H,W).cuda()
        noise = random_noise * sigma.view(-1,1,1,1)
        return noise


    def optimize_parameters(self, current_iter, tb_logger):
        
        
        # noise = self.genearte_poisson_noise()
        # self.lq = torch.clamp(self.gt+ noise.cuda(), 0, 1)


        # loss_dict = OrderedDict()
        # self.optimizer_g_filter.zero_grad()
        # preds = self.net_g(self.lq)

        # for name, module in self.net_g.named_modules():
        #     if name == 'module.filter':
        #         x, mask, v_set = module.feature_map

        # loss_filter = 0.
        # loss_filter += self.cri_pix(preds, self.gt) + 0. * sum(p.sum() for p in self.net_g.parameters())
        # # tmp = torch.ones_like(v_set)
        # # loss_filter = self.mseloss(tmp, v_set) + 0. * sum(p.sum() for p in self.net_g.parameters())
        # loss_dict['l_filter'] = loss_filter
        # loss_filter.backward()
        

        # if (current_iter-1) % 200 == 0:
        #     for name, module in self.net_g.named_modules():
        #         if name == 'module.filter':
        #             x, mask, v_set = module.feature_map
        #     if v_set != None:
        #         print(v_set[0])
    
        # self.optimizer_g_filter.step()

        # else:
            # n2n
            # original_noise = self.lq
            # new_noise = torch.clamp(self.lq+ noise.cuda(), 0, 1)



            # n2s
            # new_/input, mask = self.masker.mask(self.lq, current_iter)
            # net_output = model(net_input)
            
        

        if self.opt['train']['adv']:
            adv = self.pgd_attack(self.net_g, self.lq, self.gt)
        
        self.optimizer_g.zero_grad()
        self.optimizer_g_filter.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        loss_dict = OrderedDict()
        preds = self.net_g(self.lq)

    
        l_pix = 0.
        l_pix += self.cri_pix(preds, self.gt)
        loss_dict['l_real'] = l_pix


        if self.opt['train']['feature']:
            for name, module in self.net_g.named_modules():
                if name == 'module.feautre_to_img1':
                    preds_feature = module.feature_map
                    

            l_pix_feature = 0.
            l_pix_feature += self.cri_pix(preds_feature, self.gt)
            loss_dict['l_gaussian_0_55_feature'] = l_pix_feature


        if self.opt['train']['fq_aug']:
            fq_replaced = self.Random_frequency_replacing(preds, self.lq)
            preds_replaced = self.net_g(fq_replaced)
            l_pix_replaced = 0.
            l_pix_replaced += self.cri_pix(preds_replaced, self.gt)
            loss_dict['l_real_replaced'] = l_pix_replaced

            if self.opt['train']['feature']:
                for name, module in self.net_g.named_modules():
                    if name == 'module.feautre_to_img1':
                        preds_repalced_feature = module.feature_map

                l_pix_replaced_feature = 0.
                l_pix_replaced_feature += self.cri_pix(preds_repalced_feature, self.gt)
                loss_dict['l_gaussian_0_55_replaced_feature'] = l_pix_replaced_feature

    
        if self.opt['train']['adv']:
            l_adv = 0.
            adv_preds = self.net_g(adv)
            l_adv += self.cri_pix(adv_preds, preds)
            loss_dict['l_adv'] = l_adv
            

            if self.opt['train']['feature']:
                for name, module in self.net_g.named_modules():
                    if name == 'module.feautre_to_img1':
                        adv_preds_feature = module.feature_map
                        

                l_adv_feature = 0.
                l_adv_feature += self.cri_pix(adv_preds_feature, preds)
                loss_dict['l_adv_feature'] = l_adv_feature

            if self.opt['train']['fq_aug']:
                adv_fq_replaced = self.Random_frequency_replacing(adv_preds, adv)
                adv_preds_replaced = self.net_g(adv_fq_replaced)
                l_adv_replaced = 0.
                l_adv_replaced += self.cri_pix(adv_preds_replaced, preds)
                loss_dict['l_adv_replaced'] = l_adv_replaced
                
                if self.opt['train']['feature']:
                    for name, module in self.net_g.named_modules():
                        if name == 'module.feautre_to_img1':
                            adv_preds_repalced_feature = module.feature_map

                    l_adv_replaced_feature = 0.
                    l_adv_replaced_feature += self.cri_pix(adv_preds_repalced_feature, preds)
                    loss_dict['l_adv_replaced_feature'] = l_adv_replaced_feature

    
    

        if self.opt['train']['adv']:
            loss_adv = ( l_pix * 1./(1+self.alpha) + l_adv * self.alpha/(1+self.alpha) ) 
            if self.opt['train']['fq_aug']:
                loss_adv_replaced = ( l_pix_replaced * 1./(1+self.alpha) + l_adv_replaced * self.alpha/(1+self.alpha) ) 

                if self.opt['train']['feature']:
                    l_feature = ( l_pix_feature * 1./(1+self.alpha) + l_adv_feature * self.alpha/(1+self.alpha) )
                    l_replaced_feature =  ( l_pix_replaced_feature * 1./(1+self.alpha) + l_adv_replaced_feature * self.alpha/(1+self.alpha) )
                    l_total_feature = l_feature + l_replaced_feature
                    l_total = loss_adv + loss_adv_replaced + 0.1*l_total_feature + 0. * sum(p.sum() for p in self.net_g.parameters())
                else : 
                
                    l_total = loss_adv + loss_adv_replaced + 0. * sum(p.sum() for p in self.net_g.parameters())
            
            else:
                l_total = loss_adv + 0. * sum(p.sum() for p in self.net_g.parameters())
        else : 
            l_total = l_pix + 0. * sum(p.sum() for p in self.net_g.parameters())

        # loss_adv = ( l_pix * 1./(1+self.alpha) + l_adv * self.alpha/(1+self.alpha) ) 
        # loss_adv_replaced = ( l_pix_replaced * 1./(1+self.alpha) + l_adv_replaced * self.alpha/(1+self.alpha) ) 
        # l_total = loss_adv  + loss_adv_replaced + 0. * sum(p.sum() for p in self.net_g.parameters())


        l_total.backward()
            
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        # self.optimizer_g_filter.step()

        # replacement

        # if (current_iter-1) % 200 == 0:
        #     for name, module in self.net_g.named_modules():
        #         if name == 'module.filter':
        #             x, mask, v_set = module.feature_map
        #     if v_set != None:
        #         print(v_set[0])

        

        

        
    
                


        self.log_dict = self.reduce_loss_dict(loss_dict)


    def get_prune_rate(self):
        return self.prune_rate 

    def change_test_mode(self, mode):
        self.test_mode = mode
   
            
   


    def test(self):
        self.net_g.train()

        if self.test_mode == 'real':
            self.lq = self.lq
        elif self.test_mode == 'adv':
            self.lq = self.pgd_attack(self.net_g, self.gt, self.gt)
        elif self.test_mode =='seen_noise':
  
            B,C,H,W = self.lq.size()
            random_noise = torch.randn(B,C,H,W).cuda()
            sigma = (torch.rand(B) * 55) * 1./255
            sigma = sigma.cuda()
            noise = random_noise * sigma.view(-1,1,1,1)
            noise = noise.cuda()
            self.lq = torch.clamp(self.gt+ noise, 0, 1)
            # random_noise = torch.randn(B,C,H,W).cuda()
            # self.lq = torch.clamp(self.lq+ random_noise*0.2, 0, 1)
        elif self.test_mode =='unseen_noise':
            B,C,H,W = self.lq.size()
            random_noise = torch.randn(B,C,H,W).cuda()
            self.lq = torch.clamp(self.gt+ random_noise*(90/255), 0, 1)

        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics
        
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            if name != 'cnt':
                keys.append(name+'_'+self.test_mode)
            else : 
                keys.append(name)
            # keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
            
        # logger = get_root_logger()
        # logger.info(log_str)
        if self.opt['rank'] == 0:
            print(log_str)
        

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


    def print_values(self):
        print(self.net_g)


        # f = open("fftvalue.txt", 'w')
        # f = open("orivalue.txt", 'w')
        
        for k, v in self.net_g.named_parameters():
            if 'conv' in k and 'weight' in k:

                # print(k, torch.std_mean(v))
                
                # sys.stdout = open('fftvalue.txt', 'w')
                # print(k, v.size())
                # print(k, torch.fft.fftn(v, dim=(-1,-2)))
                # f.write(str(k) + str(torch.fft.fftn(v, dim=(-1,-2))) + "\n")
                f.write(str(k) + str(v) + "\n")
                
                # sys.stdout.close()
        # f.close()

    def get_prune_rate(self):
        return self.prune_rate 