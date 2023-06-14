# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from models.classifier_guidance_model import ClassifierGuidanceModel
from utils.degredations import build_degredation_model
from .ddim import DDIM


class MCG(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.H = build_degredation_model(cfg)
        self.cfg = cfg
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

        for ti, si in zip(reversed(ts), reversed(ss)):
            t = torch.ones(n).to(x.device).long() * ti
            s = torch.ones(n).to(x.device).long() * si
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            xt = xt.clone().to('cuda').requires_grad_(True)

            scale = 1.0

            et, x0_pred = self.model(xt, y, t, scale=scale)
            xs1 = alpha_s.sqrt() * x0_pred.detach() + c1 * torch.randn_like(xt) + c2 * et.detach()
            
            # mat = (H.H_pinv(y_0) - H.H_pinv(H.H(x0_pred))).detach().reshape(n, -1)

            # mat_x = (mat * x0_pred.reshape(n, -1)).sum()
            
            mat_x = ((H.H_pinv(y_0) - H.H_pinv(H.H(x0_pred))) ** 2).sum(dim=0).sum()
            print(mat_x)

            grad_term = torch.autograd.grad(mat_x, xt, retain_graph=True)[0] * self.grad_term_weight * alpha_t.sqrt()
            grad_term = grad_term.detach() * (1 - H.singulars().view(x0_pred.size()))

            xs2 = xs1 - grad_term
            ys = H.H_pinv(y_0).view(xt.size()) * alpha_s.sqrt() + (1 - alpha_s).sqrt() * torch.randn_like(xt)
            # x0_pred = (x0_pred + grad_term).detach()
            
            xs = xs2 * (1 - H.singulars().view(x0_pred.size())) + ys * H.singulars().view(xt.size())
            xs = xs.detach()

            # if 'in2' in self.cfg.algo.deg:
                # x0_pred = x0_pred * (1 - H.singulars().view(x0_pred.size())) + (y_0 * H.singulars()).view(x0_pred.size())

            # if not self.awd:
            #     et = (xt - x0_pred * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
            # et = et.detach()

            xt_s.append(xs.detach().cpu())
            x0_s.append(x0_pred.detach().cpu())
            xt = xs
            
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
