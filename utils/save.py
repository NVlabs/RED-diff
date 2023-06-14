import os
import lmdb
import numpy as np
import torch
import torchvision.utils as tvu
import torch.distributed as dist


def save_imagenet_result(x, y, info, samples_root, suffix=""):
        
    if len(x.shape) == 3:
        n=1
    else:
        n = x.size(0)
        
        
    for i in range(n):
        #print('info["class_id"][i]', info["class_id"][i])
        class_dir = os.path.join(samples_root, info["class_id"][i])
        #print('class_dir', class_dir)
        os.makedirs(class_dir, exist_ok=True)
    for i in range(n):
        if len(suffix) > 0:
            tvu.save_image(x[i], os.path.join(samples_root, info["class_id"][i], f'{info["name"][i]}_{suffix}.png'))
        else:
            tvu.save_image(x[i], os.path.join(samples_root, info["class_id"][i], f'{info["name"][i]}.png'))

    dist.barrier()


def save_ffhq_result(x, y, info, samples_root, suffix=""):
    x_list = [torch.zeros_like(x) for i in range(dist.get_world_size())]
    idx = info['index']
    idx_list = [torch.zeros_like(idx) for i in range(dist.get_world_size())]
    dist.gather(x, x_list, dst=0)
    dist.gather(idx, idx_list, dst=0)

    if len(suffix) == 0:
        lmdb_path = f'{samples_root}.lmdb'
    else:
        lmdb_path = f'{samples_root}_{suffix}.lmdb'

    lmdb_dir = lmdb_path.split('/')[:-1]
    if len(lmdb_dir) > 0:
        lmdb_dir = '/'.join(lmdb_dir)
        os.makedirs(lmdb_dir, exist_ok=True)

    if dist.get_rank() == 0:
        x = torch.cat(x_list, dim=0).permute(0, 2, 3, 1).detach().cpu().numpy()
        idx = torch.cat(idx_list, dim=0).detach().cpu().numpy()
        x = (x * 255.).astype(np.uint8)
        n = x.shape[0]
        env = lmdb.open(lmdb_path, map_size=int(1e12), readonly=False)
        with env.begin(write=True) as txn:
            for i in range(n):
                xi = x[i].copy()
                txn.put(str(int(idx[i])).encode(), xi)

    dist.barrier()


def save_result(name, x, y, info, samples_root, suffix=""):
    if 'ImageNet' in name:
        save_imagenet_result(x, y, info, samples_root, suffix)
    elif 'FFHQ' in name:
        save_ffhq_result(x, y, info, samples_root, suffix)
