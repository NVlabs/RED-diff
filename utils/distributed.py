import logging
import os
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig


def init_processes(rank, size, fn, cfg, cwd):
    """ Initialize the distributed environment. """
    try:
        cfg = OmegaConf.create(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg.cwd = cwd

        os.environ["MASTER_ADDR"] = cfg.dist.master_address
        os.environ["MASTER_PORT"] = str(cfg.dist.port)
        dist.init_process_group(backend=cfg.dist.backend, init_method="env://", rank=rank, world_size=size)
        fn(cfg)
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        logging.error(traceback.format_exc())
        dist.destroy_process_group()


def common_init(rank, seed):
    # we use different seeds per gpu. But we sync the weights after model initialization.
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True


def broadcast_params(params, is_distributed):
    if is_distributed:
        for param in params:
            dist.broadcast(param.data, src=0)


def get_logger(name=None, cfg=None):
    if dist.get_rank() == 0 or not dist.is_available():
        load_path = os.path.join(cfg.cwd, ".hydra/hydra.yaml")
        hydra_conf = OmegaConf.load(load_path)
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return logging.getLogger(name)


def get_results_file(cfg, logger):
    if dist.get_rank() == 0 or not dist.is_available():
        results_root = os.path.join(cfg.exp.root, 'results')
        os.makedirs(results_root, exist_ok=True)
        if '/' in cfg.results:
            results_dir = '/'.join(cfg.results.split('/')[:-1])
            results_dir = os.path.join(results_root, results_dir)
            logger.info(f'Creating directory {results_dir}')
            os.makedirs(results_dir, exist_ok=True)
        results_file = f'{results_root}/{cfg.results}.yaml'
    return results_file


def distributed(func):
    def wrapper(cfg):
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
                init_processes, args=(world_size, func, cfg, cwd), nprocs=world_size, join=True,
            )
        else:
            init_processes(0, size, func, cfg, cwd)

    return wrapper