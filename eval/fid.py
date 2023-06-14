import os

import hydra
import numpy as np
from cleanfid.fid import frechet_distance, kernel_distance
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch.distributed as dist
from utils.distributed import get_logger, init_processes, get_results_file


def main(cfg: DictConfig):
    logger = get_logger("main", cfg)
    features1 = np.load(cfg.path1)
    features2 = np.load(cfg.path2)
    
    # print('cfg.path1', cfg.path1)
    # print('features1', features1)

    if 'npy' in cfg.path1:
        mu1 = np.mean(features1, axis=0)
        sigma1 = np.cov(features1, rowvar=False)
    else:
        mu1, sigma1 = features1['mu'], features1['sigma']

    if 'npy' in cfg.path2:
        mu2 = np.mean(features2, axis=0)
        sigma2 = np.cov(features2, rowvar=False)
    else:
        mu2, sigma2 = features2['mu'], features2['sigma']

    results_file = get_results_file(cfg, logger)
    
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    logger.info(f"FID: {fid:.4f}")
    with open(results_file, 'a') as f:
        f.write(f"FID: {fid}\n")

    if 'npy' in cfg.path1 and 'npy' in cfg.path2:
        kid = kernel_distance(features1, features2)
        logger.info(f"KIDx10^3: {kid * 1000:.7f}")
        with open(results_file, 'a') as f:
            f.write(f"KID: {kid}\n")


@hydra.main(version_base="1.2", config_path="_configs", config_name="fid")
def main_dist(cfg: DictConfig):
    cwd = HydraConfig.get().runtime.output_dir
    init_processes(0, 1, main, cfg, cwd)
    #print('cfg', cfg)


if __name__ == "__main__":
    main_dist()
