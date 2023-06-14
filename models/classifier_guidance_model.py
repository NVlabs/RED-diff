import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
from .diffusion import Diffusion


class ClassifierGuidanceModel:
    def __init__(self, model: nn.Module, classifier: nn.Module, diffusion: Diffusion, cfg: DictConfig):
        self.model = model
        self.classifier = classifier
        self.diffusion = diffusion
        self.cfg = cfg

    def __call__(self, xt, y, t, scale=1.0):
        # Returns both the noise value (score function scaled) and the predicted x0.
        alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
        if self.classifier is None:
            et = self.model(xt, t)[:, :3]
        else:
            et = self.model(xt, t, y)[:, :3]
            et = et - (1 - alpha_t).sqrt() * self.cond_fn(xt, y, t, scale=scale)
        x0_pred = (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
        return et, x0_pred

    def cond_fn(self, xt, y, t, scale=1.0):
        with torch.enable_grad():
            x_in = xt.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]

            scale = scale * self.cfg.classifier.classifier_scale
            return torch.autograd.grad(selected.sum(), x_in, create_graph=True)[0] * scale