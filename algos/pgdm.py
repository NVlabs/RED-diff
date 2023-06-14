# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from models.classifier_guidance_model import ClassifierGuidanceModel
from utils.degredations import build_degredation_model
from .ddim import DDIM


class PGDM(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.H = build_degredation_model(cfg)
        self.cfg = cfg
        self.awd = cfg.algo.awd
        self.cond_awd = cfg.algo.cond_awd
        self.mcg = cfg.algo.mcg
        self.grad_term_weight = cfg.algo.grad_term_weight
        self.eta = cfg.algo.eta

    def sample(self, x, y, ts, **kwargs):
        y_0 = kwargs["y_0"]
        n = x.size(0)
        H = self.H
    
        x = self.initialize(x, y, ts, y_0=y_0)
        ss = [-1] + list(ts[:-1])
        xt_s = [x.cpu()]
        x0_s = []

        xt = x
        tot = 0
        for ti, si in zip(reversed(ts), reversed(ss)):
            t = torch.ones(n).to(x.device).long() * ti
            s = torch.ones(n).to(x.device).long() * si
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            xt = xt.clone().to('cuda').requires_grad_(True)

            if self.cond_awd:
                scale = alpha_s.sqrt() / (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt())
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0

            et, x0_pred = self.model(xt, y, t, scale=scale)
            mat = (H.H_pinv(y_0) - H.H_pinv(H.H(x0_pred))).reshape(n, -1)
            

            mat_x = (mat.detach() * x0_pred.reshape(n, -1)).sum()
            if self.cfg.algo.deg == "hdr":
                contrast = torch.std(x0_pred.view(n, -1), dim=1).sum()
                #print(mat_x, contrast)
                mat_x = mat_x + contrast * 1500

            # sigma_t = (1 - alpha_t).sqrt() / alpha_t.sqrt()
            # f = lambda x: torch.tanh(x)
            grad_term = torch.autograd.grad(mat_x, xt, retain_graph=True)[0]
            
            # g2 = (grad_term ** 2).sum().sqrt().item()

            grad_term = grad_term.detach()
            #  * alpha_t.sqrt() # * self.grad_term_weight * alpha_t.sqrt() # (sigma_t / f(sigma_t)) ** 2 * alpha_t.sqrt()

            coeff = alpha_s.sqrt() 
            if not self.awd:
                coeff = coeff - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt()
            coeff = coeff * alpha_t.sqrt() * self.grad_term_weight

            if self.mcg:
                coeff = alpha_t.sqrt() * self.grad_term_weight
            # coeff = torch.ones_like(alpha_s) * 1.0 / g2 * 5.0

            # tot += (grad_term ** 2).sum().sqrt().item() * coeff.item()

            # print(f'{(mat ** 2).sum().item():.4f}\t{((sigma_t / f(sigma_t)) ** 2 * alpha_t.sqrt()).item():.4f}\t{(grad_term ** 2).sum().sqrt().item() * coeff.item():.4f}\t{coeff.item() * alpha_t.sqrt().item():.6f}\t{g2:.4f}')
            # grad_term2 = torch.autograd.grad((mat ** 2).sum(), xt, retain_graph=True)[0] * alpha_t.sqrt() * self.grad_term_weight
            # import ipdb; ipdb.set_trace()

            # x0_pred = (x0_pred + grad_term).detach()
            x0_pred = x0_pred.detach()
            
            if 'in2' in self.cfg.algo.deg:
                x0_pred = x0_pred * (1 - H.singulars().view(x0_pred.size())) + (y_0 * H.singulars()).view(x0_pred.size())
                grad_term = grad_term * (1 - H.singulars().view(grad_term.size()))

            if not self.awd:
                et = (xt - x0_pred * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
            et = et.detach()

            xs = alpha_s.sqrt() * x0_pred + c1 * torch.randn_like(xt) + c2 * et + grad_term * coeff
            xt_s.append(xs.detach().cpu())
            x0_s.append(x0_pred.detach().cpu())
            xt = xs
        # print(f'tot: {tot:.4f}')
        return list(reversed(xt_s)), list(reversed(x0_s))

    def initialize(self, x, y, ts, **kwargs):
        y_0 = kwargs['y_0']
        H = self.H
        deg = self.cfg.algo.deg
        n = x.size(0)
        x_0 = H.H_pinv(y_0).view(*x.size()).detach()
        ti = ts[-1]
        t = torch.ones(n).to(x.device).long() * ti
        alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
        return alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)
