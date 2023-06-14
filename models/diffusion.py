import numpy as np
import torch


class Diffusion:
    def __init__(self, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2, num_diffusion_timesteps=1000, given_betas=None):
        from utils.functions import sigmoid
        if given_betas is None:
            if beta_schedule == "quad":
                betas = (
                    np.linspace(
                        beta_start**0.5,
                        beta_end**0.5,
                        num_diffusion_timesteps,
                        dtype=np.float64,
                    )
                    ** 2
                )
            elif beta_schedule == "linear":
                betas = np.linspace(
                    beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
                )
            elif beta_schedule == "const":
                betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(
                    num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
                )
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, num_diffusion_timesteps)
                betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
            else:
                raise NotImplementedError(beta_schedule)
            assert betas.shape == (num_diffusion_timesteps,)
            betas = torch.from_numpy(betas)
        else:
            betas = given_betas
        self.betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0).cuda().float()
        self.alphas = (1 - self.betas).cumprod(dim=0).cuda().float()
        self.num_diffusion_timesteps = num_diffusion_timesteps
    
    def alpha(self, t):
        return self.alphas.index_select(0, t+1)
