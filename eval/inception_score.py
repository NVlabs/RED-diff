import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

import torch_fidelity

from datasets import build_loader
from utils.distributed import get_logger, get_results_file, init_processes


class InceptionDataset(torch.utils.data.Dataset):
    def __init__(self, dset):
        super(InceptionDataset, self).__init__()
        self.dset = dset

    def __len__(self):
        return self.dset.__len__()

    def __getitem__(self, index: int):
        return self.dset[index][0]


def main(cfg: DictConfig):
    logger = get_logger("inception_score", cfg)
    loader = build_loader(cfg)

    dset = InceptionDataset(loader.dataset)
    metrics = torch_fidelity.calculate_metrics(input1=dset, cuda=True, isc=True, verbose=True, samples_find_deep=True)
    isc_mean = metrics['inception_score_mean']
    isc_std = metrics['inception_score_std']

    if dist.get_rank() == 0:
        results_file = get_results_file(cfg, logger)
        logger.info(f"IS: {isc_mean} +/- {isc_std}")

        with open(results_file, 'a') as f:
            f.write(f'IS_mean: {isc_mean}')
            f.write(f'IS_std: {isc_std}')

    dist.barrier()


@hydra.main(version_base="1.2", config_path="_configs", config_name="inception_score")
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
