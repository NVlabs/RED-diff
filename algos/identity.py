# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

from omegaconf import DictConfig

from models.classifier_guidance_model import ClassifierGuidanceModel


class Identity:
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion

    def sample(self, x, y, ts, **kwargs):
        return [x], []
