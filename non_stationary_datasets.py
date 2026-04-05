# cm_datasets.py
import os
import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset

class ClassIndexedDataset(Dataset):
    """
    Minimal wrapper that stores (x, y) with class index y in range [0..C-1].
    x is a tensor image (flattened 784 for Omniglot/MNIST-like, or CHW for others).
    """
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor):
        assert xs.size(0) == ys.size(0)
        self.xs = xs
        self.ys = ys.long()

        # build index per class
        self.class_to_indices: Dict[int, List[int]] = {}
        unique = torch.unique(self.ys).tolist()
        for c in unique:
            c = int(c)
            self.class_to_indices[c] = torch.where(self.ys == c)[0].tolist()

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    # def sample_n_way_k_shot(self, n_way: int, k_shot: int, q_query: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Returns support (Sx, Sy) and query (Qx, Qy):
    #     - Sx: (n_way*k_shot, ...) Sy: (n_way*k_shot,)
    #     - Qx: (n_way*q_query, ...) Qy: (n_way*q_query,)
    #     """
    #     classes = list(self.class_to_indices.keys())
    #     if len(classes) < n_way:
    #         raise RuntimeError(f"Dataset has only {len(classes)} classes; need {n_way}.")

    #     picked_classes = random.sample(classes, n_way)
    #     Sx_list, Sy_list, Qx_list, Qy_list = [], [], [], []

    #     for i, c in enumerate(picked_classes):
    #         idxs = self.class_to_indices[c]
    #         if len(idxs) < (k_shot + q_query):
    #             # sample with replacement if needed
    #             sup = random.choices(idxs, k=k_shot)
    #             qry = random.choices(idxs, k=q_query)
    #         else:
    #             sup = random.sample(idxs, k_shot)
    #             rest = [j for j in idxs if j not in sup]
    #             qry = random.sample(rest, q_query) if len(rest) >= q_query else random.choices(rest or idxs, k=q_query)

    #         Sx_list.append(self.xs[sup])
    #         Sy_list.append(torch.full((k_shot,), i, dtype=torch.long))  # episodic relabel 0..n_way-1
    #         Qx_list.append(self.xs[qry])
    #         Qy_list.append(torch.full((q_query,), i, dtype=torch.long))

    #     Sx = torch.cat(Sx_list, dim=0)
    #     Sy = torch.cat(Sy_list, dim=0)
    #     Qx = torch.cat(Qx_list, dim=0)
    #     Qy = torch.cat(Qy_list, dim=0)
    #     return Sx, Sy, Qx, Qy

    def sample_n_way_k_shot(self, n_way: int, k_shot: int, q_query: int):
        classes = list(self.class_to_indices.keys())
        if len(classes) < n_way:
            raise RuntimeError(f"Dataset has only {len(classes)} classes; need {n_way}.")

        picked_classes = random.sample(classes, n_way)  # global labels
        epi_to_global = {i: int(g) for i, g in enumerate(picked_classes)}

        Sx_list, Sy_list, Sg_list = [], [], []
        Qx_list, Qy_list, Qg_list = [], [], []

        for i, g in enumerate(picked_classes):
            idxs = self.class_to_indices[g]
            if len(idxs) < (k_shot + q_query):
                sup = random.choices(idxs, k=k_shot)
                qry = random.choices(idxs, k=q_query)
            else:
                sup = random.sample(idxs, k_shot)
                rest = [j for j in idxs if j not in sup]
                qry = random.sample(rest, q_query) if len(rest) >= q_query else random.choices(rest or idxs, k=q_query)

            Sx_list.append(self.xs[sup])
            Sy_list.append(torch.full((k_shot,), i, dtype=torch.long))    # episodic label
            Sg_list.append(torch.full((k_shot,), int(g), dtype=torch.long))  # global label

            Qx_list.append(self.xs[qry])
            Qy_list.append(torch.full((q_query,), i, dtype=torch.long))
            Qg_list.append(torch.full((q_query,), int(g), dtype=torch.long))

        Sx = torch.cat(Sx_list, dim=0)
        Sy = torch.cat(Sy_list, dim=0)
        Sg = torch.cat(Sg_list, dim=0)

        Qx = torch.cat(Qx_list, dim=0)
        Qy = torch.cat(Qy_list, dim=0)
        Qg = torch.cat(Qg_list, dim=0)

        return Sx, Sy, Qx, Qy, Sg, Qg


def load_ac_cml_style_dataset(root: str, name: str, split: str = "train") -> ClassIndexedDataset:
    """
    Adapt to your prepared datasets. If you already have tensors saved, load them.
    Otherwise, wrap your existing loaders to return tensors for the split.

    Expected: a file like <root>/<name>_<split>.pt -> dict with 'x' (N, ...) and 'y' (N,)
    Modify to match your actual prepare_*.py outputs.
    """
    path = os.path.join(root, f"{name}_{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset tensor file: {path}")
    blob = torch.load(path, map_location="cpu")
    xs = blob["x"]   # tensor (N, D) or (N, C, H, W)
    ys = blob["y"]   # tensor (N,)
    return ClassIndexedDataset(xs, ys)


# Cache to avoid re-loading large datasets on graph rebuilds.
_DATASET_REGISTRY_CACHE = {}

def get_dataset_registry(data_root: str, split: str = "train") -> Dict[str, ClassIndexedDataset]:
    """
    Build a registry of datasets treated as separate tasks.
    Order is fixed to: VGG-Flowers, miniImageNet, CIFAR-FS, Omniglot.
    """
    cache_key = (os.path.abspath(data_root), split)
    if cache_key in _DATASET_REGISTRY_CACHE:
        return _DATASET_REGISTRY_CACHE[cache_key]
    names = ["vggflowers", "mini_imagenet", "cifar_fs", "omniglot"]
    reg = {}
    for nm in names:
        reg[nm] = load_ac_cml_style_dataset(data_root, nm, split=split)
    _DATASET_REGISTRY_CACHE[cache_key] = (reg, names)
    return reg, names
