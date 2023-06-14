# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

from models.classifier_guidance_model import ClassifierGuidanceModel
import torch
from omegaconf import DictConfig

from models.classifier_guidance_model import ClassifierGuidanceModel
from utils.degredations import build_degredation_model
from .ddim import DDIM


class DDRM(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.eta = cfg.algo.eta
        self.eta_b = cfg.algo.eta_b
        self.sigma_y = cfg.algo.sigma_y * 2
        self.cfg = cfg
        self.H = build_degredation_model(cfg)

    @torch.no_grad()
    def sample(self, x, y, ts, **kwargs):
        x = torch.randn_like(x)
        y_0 = kwargs["y_0"]
        sigma_y = self.sigma_y
        eta = self.eta
        eta_b = self.eta_b
        n = x.size(0)

        with torch.no_grad():
            H = self.H
            # setup vectors used in the algorithm
            singulars = H.singulars()
            Sigma = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3], device=x.device)
            
            ## TODO: a hack for free form inpainting, can use batch size of 1 only.
            singulars = singulars.view(-1)
            ##

            Sigma[: singulars.shape[0]] = singulars
            U_t_y = H.Ut(y_0)
            Sig_inv_U_t_y = U_t_y / singulars[: U_t_y.shape[-1]]

            # initialize x_T as given in the paper
            # largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
            T = torch.ones(n).to(x.device).long() * ts[-1]
            largest_alphas = self.diffusion.alpha(T).view(-1, 1, 1, 1)
            largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
            large_singulars_index = torch.where(
                singulars * largest_sigmas[0, 0, 0, 0] > sigma_y
            )
            inv_singulars_and_zero = torch.zeros(
                x.shape[1] * x.shape[2] * x.shape[3]
            ).to(singulars.device)
            inv_singulars_and_zero[large_singulars_index] = (
                sigma_y / singulars[large_singulars_index]
            )
            inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

            # implement p(x_T | x_0, y) as given in the paper
            # if eigenvalue is too small, we just treat it as zero (only for init)
            init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(
                x.device
            )
            init_y[:, large_singulars_index[0]] = U_t_y[
                :, large_singulars_index[0]
            ] / singulars[large_singulars_index].view(1, -1)
            init_y = init_y.view(*x.size())
            remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero**2
            remaining_s = (
                remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                .clamp_min(0.0)
                .sqrt()
            )
            init_y = init_y + remaining_s * x
            init_y = init_y / largest_sigmas

            # setup iteration variables
            x = H.V(init_y.view(x.size(0), -1)).view(*x.size())
            n = x.size(0)
            ss = [-1] + list(ts[:-1])
            x0_s = []
            xt_s = [x]

            # iterate over the timesteps
            for ti, si in zip(reversed(ts), reversed(ss)):
                t = torch.ones(n).to(x.device).long() * ti
                s = torch.ones(n).to(x.device).long() * si
                alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
                alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
                xt = xt_s[-1].to("cuda")

                et, x0_pred = self.model(xt, y, t)

                # variational inference conditioned on y
                sigma = (1 - alpha_t).sqrt()[0, 0, 0, 0] / alpha_t.sqrt()[0, 0, 0, 0]
                sigma_s = (1 - alpha_s).sqrt()[0, 0, 0, 0] / alpha_s.sqrt()[
                    0, 0, 0, 0
                ]
                xt_mod = xt / alpha_t.sqrt()[0, 0, 0, 0]
                V_t_x = H.Vt(xt_mod)
                SVt_x = (V_t_x * Sigma)[:, : U_t_y.shape[1]]
                V_t_x0 = H.Vt(x0_pred)
                SVt_x0 = (V_t_x0 * Sigma)[:, : U_t_y.shape[1]]

                falses = torch.zeros(
                    V_t_x0.shape[1] - singulars.shape[0],
                    dtype=torch.bool,
                    device=xt.device,
                )
                cond_before_lite = singulars * sigma_s > sigma_y
                cond_after_lite = singulars * sigma_s < sigma_y
                cond_before = torch.hstack((cond_before_lite, falses))
                cond_after = torch.hstack((cond_after_lite, falses))

                std_nextC = sigma_s * eta
                sigma_tilde_nextC = torch.sqrt(sigma_s**2 - std_nextC**2)

                std_nextA = sigma_s * eta
                sigma_tilde_nextA = torch.sqrt(sigma_s**2 - std_nextA**2)

                diff_sigma_t_nextB = torch.sqrt(
                    sigma_s**2
                    - sigma_y**2 / singulars[cond_before_lite] ** 2 * (eta_b**2)
                )

                # missing pixels
                Vt_xt_mod_next = (
                    V_t_x0
                    + sigma_tilde_nextC * H.Vt(et)
                    + std_nextC * torch.randn_like(V_t_x0)
                )

                # less noisy than y (after)
                Vt_xt_mod_next[:, cond_after] = (
                    V_t_x0[:, cond_after]
                    + sigma_tilde_nextA
                    * ((U_t_y - SVt_x0) / sigma_y)[:, cond_after_lite]
                    + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
                )

                # noisier than y (before)
                Vt_xt_mod_next[:, cond_before] = (
                    Sig_inv_U_t_y[:, cond_before_lite] * eta_b
                    + (1 - eta_b) * V_t_x0[:, cond_before]
                    + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite]
                )

                # aggregate all 3 cases and give next prediction
                xt_mod_next = H.V(Vt_xt_mod_next)
                xs = (alpha_s.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

                x0_s.append(x0_pred.to("cpu"))
                xt_s.append(xs.to("cpu"))

        return list(reversed(xt_s)), list(reversed(x0_s))
