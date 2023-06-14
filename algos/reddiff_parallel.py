# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from models.classifier_guidance_model import ClassifierGuidanceModel
from utils.degredations import build_degredation_model
from .ddim import DDIM
import random


class REDDIFF_PARALLEL(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.H = build_degredation_model(cfg)
        self.cfg = cfg
        self.awd = cfg.algo.awd
        self.cond_awd = cfg.algo.cond_awd
        self.grad_term_weight = cfg.algo.grad_term_weight
        self.eta = cfg.algo.eta
        self.lr = cfg.algo.lr
        self.denoise_term_weight = cfg.algo.denoise_term_weight

        #assert cfg.algo.deg == 'deno' or 'inp' in cfg.algo.deg 

    def sample(self, x, y, ts, **kwargs):
        y_0 = kwargs["y_0"]
        sigma_y = self.cfg.algo.sigma_y
        n = x.size(0)
        H = self.H
    
        x = self.initialize(x, y, ts, y_0=y_0)
        ss = [-1] + list(ts[:-1])
        xt_s = [x.cpu()]
        x0_s = []
        
        #optimizer
        dtype = torch.FloatTensor
        mu = torch.autograd.Variable(x, requires_grad=True)       #device=device).type(dtype)
        optimizer = torch.optim.Adam([mu], lr=1e-1, betas=(0.9, 0.99), weight_decay=0.0)     #original beta2: 0.999, lr=1e-1
        #optimizer = torch.optim.SGD([mu], lr=1e6, momentum=0.9)  #momentum=0.9
        
        num_dens = 1  #10   #25
        num_iter = self.cfg.exp.num_steps // num_dens
        ts = reversed(ts)
        ss = reversed(ss)
        temp = list(zip(ts, ss))
        #random.shuffle(temp)
        ts, ss = zip(*temp)
        ts_par = [ts[i*num_dens:(i+1)*num_dens] for i in range(num_iter)]*1
        ss_par = [ss[i*num_dens:(i+1)*num_dens] for i in range(num_iter)]*1
        
        
        for ti, si in zip(ts_par, ss_par):
            
            t = torch.ones(n,1).to(x.device).long() * torch.tensor(ti).to(x.device).long()   #bs x num_dens
            s = torch.ones(n,1).to(x.device).long() * torch.tensor(si).to(x.device).long()
            t = t.view(1,-1)[0]
            s = s.view(1,-1)[0]
            
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            
            sigma_x0 = 0.0001
            noise_x0 = torch.randn_like(mu)
            x0_pred = mu + sigma_x0*noise_x0
            
            mu_repeat = mu.repeat(num_dens,1,1,1)
            noise_x0_repeat = noise_x0.repeat(num_dens,1,1,1)
            x0_pred_repeat = x0_pred.repeat(num_dens,1,1,1)

            noise_xt = torch.randn_like(mu_repeat)
                        
            xt = alpha_t.sqrt() * x0_pred_repeat + (1 - alpha_t).sqrt() * noise_xt
                        
            #scale = 0.0
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            #xt = xt.clone().to('cuda').requires_grad_(True)
            if self.cond_awd:
                scale = alpha_s.sqrt() / (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt())
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0

            yt = y.repeat(num_dens)
            
            et, x0_hat = self.model(xt, yt, t, scale=scale)   #et, x0_pred
                        
            if not self.awd:
                et = (xt - x0_hat * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
            et = et.detach()
            
            
            snr_inv = (1-alpha_t).sqrt()/alpha_t.sqrt()  #1d torch tensor
            
            if self.denoise_term_weight == "linear":
                snr_inv = snr_inv
            elif self.denoise_term_weight == "sqrt":
                snr_inv = torch.sqrt(snr_inv)
            elif self.denoise_term_weight == "square":
                snr_inv = torch.square(snr_inv)
            elif self.denoise_term_weight == "log":
                snr_inv = torch.log(snr_inv + 1.0)
            elif self.denoise_term_weight == "trunc_linear":
                snr_inv = torch.clip(snr_inv, max=1.0)
            elif self.denoise_term_weight == "power2over3":
                snr_inv = torch.pow(snr_inv, 2/3)
            elif self.denoise_term_weight == "const":
                snr_inv = torch.pow(snr_inv, 0.0)
            
            if 'in' in self.cfg.algo.deg:
                #inpaint + adam
                w_t = self.grad_term_weight*snr_inv
                v_t = 1.0
            elif 'sr' in self.cfg.algo.deg:
                #sr + adam
                w_t = self.grad_term_weight*snr_inv
                v_t = 1.0
            
            e_obs = y_0 - H.H(x0_pred)
            loss_obs = (e_obs**2).mean()/2
            loss_noise = torch.mul(w_t*(et - noise_xt).detach(), x0_pred_repeat).mean()

            loss = loss_noise + v_t*loss_obs
            
            #adam step
            optimizer.zero_grad()  #initialize
            loss.backward()
            optimizer.step()
        
        return x0_pred, mu  #x0_pred, mu   #list(reversed(xt_s)), list(reversed(x0_s))

    def initialize(self, x, y, ts, **kwargs):
        deg = self.cfg.algo.deg
        y_0 = kwargs['y_0']
        H = self.H
        n = x.size(0)
        ti = ts[-1]
        x_0 = H.H_pinv(y_0).view(*x.size()).detach()
        t = torch.ones(n).to(x.device).long() * ti
        alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)  #it is zero
        return x_0  #alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)    #x_0







