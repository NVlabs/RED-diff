# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
from omegaconf import DictConfig

from models.classifier_guidance_model import ClassifierGuidanceModel


class DEIS:
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.eta = cfg.algo.eta
        self.sdedit = cfg.algo.sdedit
        self.cond_awd = cfg.algo.cond_awd

    @torch.no_grad()
    def sample(self, x, y, ts, **kwargs):
        x = self.initialize(x, y, ts, **kwargs)
        n = x.size(0)
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
            if self.cond_awd:
                scale = alpha_s.sqrt() / (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt())
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0
            et, x0_pred = self.model(xt, y, t, scale=scale)
            xs = alpha_s.sqrt() * x0_pred + c1 * torch.randn_like(xt) + c2 * et
            xt_s.append(xs.cpu())
            x0_s.append(x0_pred.cpu())
            xt = xs
            
        return list(reversed(xt_s)), list(reversed(x0_s))

    def get_coef(self, ts, device):
        ss = [-1] + list(ts[:-1])
        rev_ts = list(reversed(ts))
        rev_ss = list(reversed(ss))

        rev_ts_th = torch.tensor(rev_ts).float().to(device)
        rev_ss_th = torch.tensor(rev_ss).float().to(device)

        alpha_rev_ts = self.diffusion.alpha(rev_ts_th)
        alpha_rev_ss = self.diffusion.alpha(rev_ss_th)
        psi = alpha_rev_ss / alpha_rev_ts

        


    
    def initialize(self, x, y, ts, **kwargs):
        if self.sdedit:
            n = x.size(0)
            ti = ts[-1]
            t = torch.ones(n).to(x.device).long() * ti
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            return x * alpha_t.sqrt() + torch.randn_like(x) * (1 - alpha_t).sqrt()
        else:
            return torch.randn_like(x)
