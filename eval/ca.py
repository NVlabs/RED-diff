import os

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from cleanfid.features import build_feature_extractor
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import build_loader
import torchvision.transforms as transforms
from utils.distributed import get_logger, init_processes, get_results_file

from utils.functions import accuracy


import sys
import torchvision

# print(sys.version)
# print(torch.__version__)
# print(torchvision.__version__)


def main(cfg: DictConfig):
    torch.hub.set_dir(os.path.join(cfg.exp.root, 'hub'))
    torch.cuda.set_device(dist.get_rank())
    logger = get_logger("ca", cfg)
    exp_root = os.path.join(cfg.exp.root, "fid_stats")
    os.makedirs(exp_root, exist_ok=True)
    loader = build_loader(cfg)
    model = torch.hub.load("pytorch/vision:v0.13.1", "resnet50", weights='IMAGENET1K_V1').cuda()    #, force_reload=True
    model = DDP(model, device_ids=[dist.get_rank()], output_device=[dist.get_rank()])
    model.eval()
    top1, top5 = 0, 0
    count = 0
    logger.info(f'A total of {len(loader.dataset)} images are processed.')
    for x, y, info in tqdm(loader):
        n, c, h, w = x.size()
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            y_ = model(x)
            t1, t5 = accuracy(y_, y, topk=(1, 5))
            top1 += t1 * n
            top5 += t5 * n
            count += n
    
    features = torch.tensor([top1, top5, count]).cpu()
    features_list = [torch.zeros_like(features) for i in range(dist.get_world_size())]
    dist.gather(features, features_list, dst=0)

    if dist.get_rank() == 0:
        features = torch.stack(features_list, dim=1)
        top1_tot = torch.sum(features[0], dim=0).item()
        top5_tot = torch.sum(features[1], dim=0).item()
        count_tot = torch.sum(features[2], dim=0).item()

        logger.info(f"Top1: {top1_tot / count_tot}, Top5: {top5_tot / count_tot}.")
        results_file = get_results_file(cfg, logger)

        with open(results_file, 'a') as f:
            f.write(f"Total: {count_tot}\nTop1: {top1_tot / count_tot}\nTop5: {top5_tot / count_tot}\n")

    dist.barrier()


@hydra.main(version_base="1.2", config_path="_configs", config_name="ca")
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
