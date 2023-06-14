# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import datetime
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb
import torchvision.utils as tvu
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algos import build_algo
from datasets import build_loader
from models import build_model
from models.classifier_guidance_model import ClassifierGuidanceModel
from models.diffusion import Diffusion
from utils.distributed import get_logger, init_processes, common_init
from utils.functions import get_timesteps, postprocess, preprocess, strfdt
from utils.degredations import get_degreadation_image
from utils.save import save_result

torch.set_printoptions(sci_mode=False)

# import sys
# sys.path.append('/lustre/fsw/nvresearch/mmardani/source/latent-diffusion-sampling/pgdm')
# print(sys.path)
#import pdb; pdb.set_trace()


def main(cfg):
    print('cfg.exp.seed', cfg.exp.seed)
    common_init(dist.get_rank(), seed=cfg.exp.seed)
    torch.cuda.set_device(dist.get_rank())
    
    # import pdb; pdb.set_trace()
    
    logger = get_logger(name="main", cfg=cfg)
    logger.info(f'Experiment name is {cfg.exp.name}')
    exp_root = cfg.exp.root
    samples_root = cfg.exp.samples_root
    exp_name = cfg.exp.name
    samples_root = os.path.join(exp_root, samples_root, exp_name)
    dataset_name = cfg.dataset.name
    if dist.get_rank() == 0:
        if cfg.exp.overwrite:
            if os.path.exists(samples_root):
                shutil.rmtree(samples_root)
            os.makedirs(samples_root)
        else:
            if not os.path.exists(samples_root):
                os.makedirs(samples_root)
    

            
    model, classifier = build_model(cfg)
    model.eval()
    if classifier is not None:
        classifier.eval()
    loader = build_loader(cfg)
    logger.info(f'Dataset size is {len(loader.dataset)}')
    diffusion = Diffusion(**cfg.diffusion)
    cg_model = ClassifierGuidanceModel(model, classifier, diffusion, cfg)   #?? what is the easiest way to call stable diffusion?

    algo = build_algo(cg_model, cfg)
    if "ddrm" in cfg.algo.name or "mcg" in cfg.algo.name or "dps" in cfg.algo.name or "pgdm" in cfg.algo.name or "reddiff" in cfg.algo.name:
        H = algo.H

    psnrs = []
    start_time = time.time()
    for it, (x, y, info) in enumerate(loader):
        
        
        if cfg.exp.smoke_test > 0 and it >= cfg.exp.smoke_test:
            break
        

        n, c, h, w = x.size()
        x, y = x.cuda(), y.cuda()

        x = preprocess(x)
        ts = get_timesteps(cfg)

        kwargs = info
        if "ddrm" in cfg.algo.name or "mcg" in cfg.algo.name or "dps" in cfg.algo.name or "pgdm" in cfg.algo.name or "reddiff" in cfg.algo.name:
            idx = info['index']
            if 'inp' in cfg.algo.deg or 'in2' in cfg.algo.deg:   #what is in2?
                H.set_indices(idx)
            y_0 = H.H(x)

            # This is to account for scaling to [-1, 1]
            y_0 = y_0 + torch.randn_like(y_0) * cfg.algo.sigma_y * 2    #?? what is it for???
            kwargs["y_0"] = y_0
        
        
        #pgdm
        if cfg.exp.save_evolution:
            xt_s, _, xt_vis, _, mu_fft_abs_s, mu_fft_ang_s = algo.sample(x, y, ts, **kwargs)   
        else:    
            xt_s, _ = algo.sample(x, y, ts, **kwargs)  
        
        #visualiztion of steps
        if cfg.exp.save_evolution:
            xt_vis = postprocess(xt_vis).cpu()
            print('torch.max(mu_fft_abs_s)', torch.max(mu_fft_abs_s))
            print('torch.min(mu_fft_abs_s)', torch.min(mu_fft_abs_s))
            print('torch.max(mu_fft_ang_s)', torch.max(mu_fft_ang_s))
            print('torch.min(mu_fft_ang_s)', torch.min(mu_fft_ang_s))
            mu_fft_abs = torch.log(mu_fft_abs_s+1)
            mu_fft_ang = mu_fft_ang_s  #torch.log10(mu_fft_abs_s+1)
            mu_fft_abs = (mu_fft_abs - torch.min(mu_fft_abs))/(torch.max(mu_fft_abs) - torch.min(mu_fft_abs))
            mu_fft_ang = (mu_fft_ang - torch.min(mu_fft_ang))/(torch.max(mu_fft_ang) - torch.min(mu_fft_ang))
            xx = torch.cat((xt_vis, mu_fft_abs, mu_fft_ang), dim=2)
            save_result(dataset_name, xx, y, info, samples_root, "evol")
                    
        #timing
        # start_time_sample = time.time()
        # finish_time_sample = time.time() - start_time
        # print('cfg.loader.batch_size', cfg.loader.batch_size)
        # print('cfg.exp.num_steps', cfg.exp.num_steps)
        # time_per_sample = finish_time_sample/(cfg.exp.num_steps*cfg.loader.batch_size)
        # print('time_per_sample', time_per_sample)
        # import pdb; pdb.set_trace()
        
        
        if isinstance(xt_s, list):
            xo = postprocess(xt_s[0]).cpu()
        else:
            xo = postprocess(xt_s).cpu()
        
        save_result(dataset_name, xo, y, info, samples_root, "")
        
        mse = torch.mean((xo - postprocess(x).cpu()) ** 2, dim=(1, 2, 3))
        psnr = 10 * torch.log10(1 / (mse + 1e-10))
        psnrs.append(psnr)

        if cfg.exp.save_deg:
            xo = postprocess(get_degreadation_image(y_0, H, cfg))

            save_result(dataset_name, xo, y, info, samples_root, "deg")
        
        if cfg.exp.save_ori:
            xo = postprocess(x)
            save_result(dataset_name, xo, y, info, samples_root, "ori")

        if it % cfg.exp.logfreq == 0 or cfg.exp.smoke_test > 0 or it < 10:
            now = time.time() - start_time
            now_in_hours = strfdt(datetime.timedelta(seconds=now))
            future = (len(loader) - it - 1) / (it + 1) * now
            future_in_hours = strfdt(datetime.timedelta(seconds=future))
            logger.info(f"Iter {it}: {now_in_hours} has passed, expect to finish in {future_in_hours}")
            
            
    if len(loader) > 0:
        psnrs = torch.cat(psnrs, dim=0)
        logger.info(f'PSNR: {psnrs.mean().item()}')
        
    logger.info("Done.")
    now = time.time() - start_time
    now_in_hours = strfdt(datetime.timedelta(seconds=now))
    logger.info(f"Total time: {now_in_hours}")


@hydra.main(version_base="1.2", config_path="_configs", config_name="ddrmpp")
def main_dist(cfg: DictConfig):
    cwd = HydraConfig.get().runtime.output_dir

    if cfg.dist.num_processes_per_node < 0:
        size = torch.cuda.device_count()
        cfg.dist.num_processes_per_node = size
    else:
        size = cfg.dist.num_processes_per_node
    if size > 1:
        num_proc_node = cfg.dist.num_proc_node
        num_process_per_node = cfg.dist.num_processes_per_node
        world_size = num_proc_node * num_process_per_node
        mp.spawn(
            init_processes, args=(world_size, main, cfg, cwd), nprocs=world_size, join=True,
        )
    else:
        init_processes(0, size, main, cfg, cwd)


if __name__ == "__main__":
    main_dist()
