import os
import platform

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from datasets import build_loader
from utils.distributed import get_logger, init_processes

from cleanfid.fid import get_batch_features
from cleanfid.downloads_helper import check_download_url
from cleanfid.inception_pytorch import InceptionV3
from cleanfid.inception_torchscript import InceptionV3W


"""
returns a functions that takes an image in range [0,255]
and outputs a feature embedding vector
"""
def feature_extractor(name="torchscript_inception", device=torch.device("cuda"), resize_inside=False):
    if name == "torchscript_inception":
        path = "./" if platform.system() == "Windows" else "/tmp"
        model = InceptionV3W(path, download=True, resize_inside=resize_inside).to(device)
        model.eval()
        def model_fn(x): return model(x)
    elif name == "pytorch_inception":
        model = InceptionV3(output_blocks=[3], resize_input=False).to(device)
        model.eval()
        def model_fn(x): return model(x/255)[0].squeeze(-1).squeeze(-1)
    else:
        raise ValueError(f"{name} feature extractor not implemented")
    return model_fn


"""
Build a feature extractor for each of the modes
"""
def build_feature_extractor(mode, device=torch.device("cuda")):
    if mode == "legacy_pytorch":
        feat_model = feature_extractor(name="pytorch_inception", resize_inside=False, device=device)
    elif mode == "legacy_tensorflow":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=True, device=device)
    elif mode == "clean":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=False, device=device)
    return feat_model


# def get_batch_features(batch, model, device):
#     with torch.no_grad():
#         feat = model(batch.to(device))
#     return feat.detach().cpu().numpy()


class ToNumpy:
    def __call__(self, img):
        return img.cpu().numpy()


class Resizer:
    def __init__(self, mode) -> None:
        self.mode = mode
        if mode == "clean":
            self.library = "PIL"
            self.quantize_after = False
            self.filter = "bicubic"
            self.output_size = (299, 299)
        elif mode == "legacy_pytorch":
            self.library = "PyTorch"
            self.quantize_after = False
            self.filter = "bilinear"
            self.output_size = (299, 299)
        else:
            raise ValueError(f"Invalid mode {mode} specified.")

    def __call__(self, x):
        library = self.library
        quantize_after = self.quantize_after
        filter = self.filter
        output_size = self.output_size

        if library == "PIL" and quantize_after:
            name_to_filter = {
                "bicubic": Image.BICUBIC,
                "bilinear": Image.BILINEAR,
                "nearest": Image.NEAREST,
                "lanczos": Image.LANCZOS,
                "box": Image.BOX,
            }
            x = Image.fromarray(x)
            x = x.resize(output_size, resample=name_to_filter[filter])
            x = np.asarray(x).clip(0, 255).astype(np.uint8)
            return x
        elif library == "PIL" and not quantize_after:
            name_to_filter = {
                "bicubic": Image.BICUBIC,
                "bilinear": Image.BILINEAR,
                "nearest": Image.NEAREST,
                "lanczos": Image.LANCZOS,
                "box": Image.BOX,
            }
            s1, s2 = output_size

            x_final = []
            for idx in range(3):
                x_np = x[:, :, idx]
                img = Image.fromarray(x_np.astype(np.float32), mode="F")
                img = img.resize(output_size, resample=name_to_filter[filter])
                x_final.append(np.asarray(img).clip(0, 255).reshape(s1, s2, 1))
            x = x_final
            x = np.concatenate(x, axis=2).astype(np.float32)

            return x
        elif library == "PyTorch":
            x = torch.Tensor(x)[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)

            if quantize_after:
                x = x.astype(np.uint8)
            return x
        else:
            raise ValueError()


def get_loader_features(loader, model=None, device=torch.device("cuda"), mode="clean", verbose=True):
    loader.dataset.transform = Compose([loader.dataset.transform] + [ToNumpy(), Resizer(mode), ToTensor()])
    world_size = dist.get_world_size()

    model = build_feature_extractor(mode, device)
    l_feats = []
    if verbose:
        pbar = tqdm(loader, total=len(loader) * loader.batch_size * world_size)
    else:
        pbar = loader

    for batch in pbar:
        l_feats.append(get_batch_features(batch[0], model, device).astype(np.float64))
        if verbose:
            pbar.update(loader.batch_size * world_size)
    np_feats = np.concatenate(l_feats)
    return np_feats


def get_moments(loader, model=None, device=torch.device("cuda"), mode="clean", verbose=True):
    loader.dataset.transform = Compose([loader.dataset.transform] + [ToNumpy(), Resizer(mode), ToTensor()])
    world_size = dist.get_world_size()

    model = build_feature_extractor(mode, device)
    moment1 = None
    moment2 = None
    size = 0
    if verbose:
        pbar = tqdm(loader, total=len(loader) * loader.batch_size * world_size)
    else:
        pbar = loader

    for batch in pbar:
        features = get_batch_features(batch[0], model, device).astype(np.float64)
        if moment1 is None:
            moment1 = np.mean(features, axis=0)
            moment2 = np.cov(features, rowvar=False, ddof=0) + (moment1.reshape([-1, 1]) @ moment1.reshape([1, -1]))
        else:
            m1 = np.mean(features, axis=0)
            m2 = np.cov(features, rowvar=False, ddof=0) + (m1.reshape([-1, 1]) @ m1.reshape([1, -1]))
            moment1 = (moment1 * size + m1 * features.shape[0]) / (size + features.shape[0])
            moment2 = (moment2 * size + m2 * features.shape[0]) / (size + features.shape[0])
        size = size + features.shape[0]
        if verbose:
            pbar.update(loader.batch_size * world_size)
    return moment1, moment2, size


def main(cfg: DictConfig):
    torch.hub.set_dir(os.path.join(cfg.exp.root, 'hub'))

    logger = get_logger("fid", cfg)
    fid_root = os.path.join(cfg.exp.root, "fid_stats")
    os.makedirs(fid_root, exist_ok=True)
    loader = build_loader(cfg)
    
    # print('cfg', cfg)
    # import pdb; pdb.set_trace()

    device = torch.device(f"cuda:{dist.get_rank()}")

    if cfg.mean_std_stats == False:
        if '/' in cfg.save_path:
            save_dir = '/'.join(cfg.save_path.split('/')[:-1])
            save_dir = os.path.join(fid_root, save_dir)
            os.makedirs(save_dir, exist_ok=True)
        save_path = f"{fid_root}/{cfg.save_path}.npy"
    else:
        if '/' in cfg.save_path:
            save_dir = '/'.join(cfg.save_path.split('/')[:-1])
            save_dir = os.path.join(fid_root, save_dir)
            os.makedirs(save_dir, exist_ok=True)
        save_path = f"{fid_root}/{cfg.save_path}_mean_std.npz"

    if os.path.exists(save_path):
        logger.info(f'Stats already exists for file {save_path}.')
        return

    if cfg.mean_std_stats == False:
        logger.info('No mean/std stats. Memory inefficient but can be used for KID.')
        features = get_loader_features(loader, device=device, mode=cfg.fid.mode, verbose=(dist.get_rank() == 0))
        features = torch.from_numpy(features)

        features_list = [torch.zeros_like(features) for i in range(dist.get_world_size())]
        dist.gather(features, features_list, dst=0)

        if dist.get_rank() == 0:
            features = torch.stack(features_list, dim=1)
            # TODO: check if this is correct
            features = features.reshape(-1, features.size(-1))
            features_npy = features.numpy()
            np.save(save_path, features_npy)
            logger.info(f"Save an array of shape {features_npy.shape} to {save_path}")
    else:
        logger.info('Only save mean/std stats. Memory efficient, but can only be used for FID.')
        moment1, moment2, size = get_moments(loader, device=device, mode=cfg.fid.mode, verbose=(dist.get_rank() == 0))
        mu = moment1
        sigma = moment2 - moment1.reshape([-1, 1]) @ moment1.reshape([1, -1])
        sigma = sigma * size / (size - 1)

        # TODO: dist version.
        # features = torch.from_numpy(features)

        # features_list = [torch.zeros_like(features) for i in range(dist.get_world_size())]
        # dist.gather(features, features_list, dst=0)

        if dist.get_rank() == 0:
            np.savez(save_path, mu=mu, sigma=sigma)
            logger.info(f"Save two arrays of shape {mu.shape} to {save_path}")


    dist.barrier()


@hydra.main(version_base="1.2", config_path="_configs", config_name="fid_stats")
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
        
    #print('cfg', cfg)


if __name__ == "__main__":
    main_dist()
