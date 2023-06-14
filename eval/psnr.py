import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure     #StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets import build_loader
from utils.distributed import get_logger, get_results_file, init_processes


def main(cfg: DictConfig):
    logger = get_logger("psnr", cfg)
    loader = build_loader(cfg, ["dataset1", "dataset2"])
    psnrs = []
    ssims = []
    lpips = []

    for b1, b2 in tqdm(loader):
        x1, x2 = b1[0], b2[0]
        x1 = x1.cuda()
        x2 = x2.cuda()
        mse = torch.mean((x1 - x2) ** 2, dim=(1, 2, 3))
        psnr = 10 * torch.log10(1 / (mse + 1e-10))
        ssim = structural_similarity_index_measure(x2, x1, reduction=None)
        with torch.no_grad():
            lpip = LearnedPerceptualImagePatchSimilarity().cuda()(x2, x1)
        psnrs.append(psnr)
        ssims.append(ssim)
        lpips.append(lpip.item())

    
    psnrs = torch.cat(psnrs, dim=0)
    ssims = torch.cat(ssims, dim=0)
    lpips = torch.tensor(lpips).cuda()

    psnrs_list = [torch.zeros_like(psnrs) for i in range(dist.get_world_size())]
    ssims_list = [torch.zeros_like(ssims) for i in range(dist.get_world_size())]
    lpips_list = [torch.zeros_like(lpips) for i in range(dist.get_world_size())]
    dist.gather(psnrs, psnrs_list, dst=0)
    dist.gather(ssims, ssims_list, dst=0)
    dist.gather(lpips, lpips_list, dst=0)

    if dist.get_rank() == 0:
        results_file = get_results_file(cfg, logger)
        psnrs = torch.cat(psnrs_list, dim=0)
        ssims = torch.cat(ssims_list, dim=0)
        lpips = torch.cat(lpips_list, dim=0)
        logger.info(f"PSNR: {psnrs.mean().item()} +/- {psnrs.std().item()}")
        logger.info(f"SSIM: {ssims.mean().item()}")
        logger.info(f"LPIPS: {lpips.mean().item()}")

        with open(results_file, 'a') as f:
            f.write(f'PSNR: {psnrs.mean().item()}\n')
            f.write(f'SSIM: {ssims.mean().item()}\n')
            f.write(f"LPIPS: {lpips.mean().item()}")

    dist.barrier()


@hydra.main(version_base="1.2", config_path="_configs", config_name="psnr")
def main_dist(cfg: DictConfig):
    cwd = HydraConfig.get().runtime.output_dir
    
    #print('cfg', cfg)
    #import pdb; pdb.set_trace()

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
