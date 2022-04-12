import itertools
from typing import Tuple, List
import hashlib
import random
import os
import gc
from scipy.sparse import diags
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from termcolor import cprint

from itertools import islice





# GPU

def get_gpu_utility(gpu_id_or_ids: int or list) -> List[int]:

    if isinstance(gpu_id_or_ids, int):
        gpu_ids = [gpu_id_or_ids]
    else:
        gpu_ids = gpu_id_or_ids
    return gpu_ids
    #
    # import subprocess
    # sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # out_str = sp.communicate()
    # out_list = out_str[0].decode("utf-8").split("\n")
    #
    # gpu_utilities = []
    # for item in out_list:
    #     items = [x.strip() for x in item.split(':')]
    #     if len(items) == 2:
    #         key, val = items
    #         if key == "Minor Number":
    #             gpu_utilities.append(int(val))
    #
    # gpu_utilities = [g - min(gpu_utilities) for g in gpu_utilities]
    #
    # if len(gpu_utilities) < len(gpu_ids):
    #     raise EnvironmentError(
    #         "Cannot find all GPUs whose ids are {}, only found {} GPUs".format(gpu_ids, len(gpu_utilities)))
    # else:
    #     return gpu_utilities


def get_free_gpu_names(num_gpus_total: int, threshold=30) -> List[str]:
    """
    :param num_gpus_total: total number of gpus
    :param threshold: Return GPUs the utilities of which is smaller than threshold.
    :return e.g. ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']
    """
    gpu_ids = list(range(num_gpus_total))
    gpu_utilities = get_gpu_utility(gpu_ids)
    return ["/device:GPU:{}".format(gid) for gid, utility in zip(gpu_ids, gpu_utilities) if utility <= threshold]


def get_free_gpu_names_safe(num_gpus_total: int, threshold=30, iteration=3) -> List[str]:
    gpu_set = set(get_free_gpu_names(num_gpus_total, threshold))
    for _ in range(iteration - 1):
        gpu_set = gpu_set.intersection(set(get_free_gpu_names(num_gpus_total, threshold)))
    return list(gpu_set)


def get_free_gpu_ids(num_gpus_total: int, threshold=30) -> List[int]:
    free_gpu_names = get_free_gpu_names(num_gpus_total=num_gpus_total, threshold=threshold)
    return [int(g.split(":")[-1]) for g in free_gpu_names]


def get_free_gpu_ids_safe(num_gpus_total: int, threshold=30) -> List[int]:
    free_gpu_names = get_free_gpu_names_safe(num_gpus_total=num_gpus_total, threshold=threshold)
    return [int(g.split(":")[-1]) for g in free_gpu_names]


def blind_other_gpus(num_gpus_total, num_gpus_to_use, is_safe=True, black_list=None, **kwargs):
    if is_safe:
        free_gpu_ids = get_free_gpu_ids_safe(num_gpus_total, **kwargs)
    else:
        free_gpu_ids = get_free_gpu_ids(num_gpus_total, **kwargs)

    if black_list is not None:
        free_gpu_ids = [g for g in free_gpu_ids if g not in black_list]

    if free_gpu_ids:
        gpu_ids_to_use = random.sample(free_gpu_ids, num_gpus_to_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(n) for n in gpu_ids_to_use)
    else:
        gpu_ids_to_use = []

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(num_gpus_to_use)
    return gpu_ids_to_use


# Others



def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A
