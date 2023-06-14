# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

from .ddim import DDIM
from .ddrm import DDRM
from .pgdm import PGDM
from .identity import Identity
from .reddiff import REDDIFF
from .reddiff_parallel import REDDIFF_PARALLEL
from .mcg import MCG
from .dps import DPS
from .sds import SDS
from .sds_var import SDS_VAR


def build_algo(cg_model, cfg):
    if cfg.algo.name == 'identity':
        return Identity(cg_model, cfg)
    elif cfg.algo.name == 'ddim':
        return DDIM(cg_model, cfg)
    elif cfg.algo.name == 'ddrm':
        return DDRM(cg_model, cfg)
    elif cfg.algo.name == 'pgdm':
        return PGDM(cg_model, cfg)
    elif cfg.algo.name == 'reddiff':
        return REDDIFF(cg_model, cfg)
    elif cfg.algo.name == 'reddiff_parallel':
        return REDDIFF_PARALLEL(cg_model, cfg)
    elif cfg.algo.name == 'mcg':
        return MCG(cg_model, cfg)
    elif cfg.algo.name == 'dps':
        return DPS(cg_model, cfg)
    elif cfg.algo.name == 'sds':
        return SDS(cg_model, cfg)
    elif cfg.algo.name == 'sds_var':
        return SDS_VAR(cg_model, cfg)
    else:
        raise ValueError(f'No algorithm named {cfg.algo.name}')
