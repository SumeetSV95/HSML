# train_ftml.py — FTML (online) with global buffer cap + fixed task order
# Deterministic multi-run support:
#   - Fixed data seed → per-task permutations generated once at init (training tasks only)
#   - Separate model seed controls all training-time RNG (sampling, eviction, etc.)
#   - CUDA determinism toggled where available

import argparse, random, os, json
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# for plots in headless envs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

try:
    from non_stationary_datasets import get_dataset_registry
except ImportError:  # pragma: no cover - optional dependency for new mode
    get_dataset_registry = None  # type: ignore

# ==============================
# Data loading: .pt (preferred) or .pkl fallback
# ==============================
def _to_tensor(a):
    if isinstance(a, torch.Tensor): return a
    if isinstance(a, np.ndarray):   return torch.from_numpy(a)
    return torch.tensor(a)

def _coerce_xy(x, y):
    x = _to_tensor(x).float()
    y = _to_tensor(y).long().view(-1)
    # flatten to [N,784]
    if x.dim() == 4 and x.shape[1:] == (1, 28, 28):
        x = x.view(x.size(0), -1)
    elif x.dim() == 3 and x.shape[1:] == (28, 28):
        x = x.view(x.size(0), -1)
    elif x.dim() == 2 and x.shape[1] == 784:
        pass
    else:
        raise ValueError(f"Unexpected x shape {tuple(x.shape)}")
    if x.size(0) != y.size(0):
        raise ValueError(f"x/y length mismatch: {x.size(0)} vs {y.size(0)}")
    return x, y

def load_tasks_pt(pt_path: str):
    train_raw, test_raw = torch.load(pt_path, map_location="cpu")
    def to_tasks(raw):
        tasks = []
        for idx, item in enumerate(raw):
            try:
                angle = float(item[0])
            except (TypeError, ValueError):
                # e.g., permutation MNIST uses string tags like "random permutation"
                angle = float(idx)
            X, Y  = _coerce_xy(item[1], item[2])
            tasks.append({"x": X, "y": Y, "angle": angle})
        return tasks
    train_tasks = to_tasks(train_raw)
    test_tasks  = to_tasks(test_raw)
    print(f"Parsed {len(train_tasks)} train / {len(test_tasks)} test tasks from {os.path.basename(pt_path)}")
    return train_tasks, test_tasks

def smart_load(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

def load_tasks_pkl(pkl_path: str) -> List[Dict[str, torch.Tensor]]:
    obj = smart_load(pkl_path)

    def _extract_xy_from_block(block: Dict[str, Any]):
        for kx, ky in (("x","y"), ("images","labels"), ("X","Y")):
            if kx in block and ky in block:
                return _coerce_xy(block[kx], block[ky])
        if isinstance(block, (tuple, list)) and len(block) == 2:
            return _coerce_xy(block[0], block[1])
        raise KeyError("no x/y in block")

    def _try_support_query_dict(val: Dict[str, Any]):
        if "support" in val and "query" in val:
            sx, sy = _extract_xy_from_block(val["support"])
            qx, qy = _extract_xy_from_block(val["query"])
            return torch.cat([sx, qx], 0), torch.cat([sy, qy], 0)

    def _try_prefixed_keys(val: Dict[str, Any]):
        def get_any(d, names):
            for n in names:
                if n in d: return d[n]
            return None
        sx = get_any(val, ["support_x","support_images","sx","supportX"])
        sy = get_any(val, ["support_y","support_labels","sy","supportY"])
        qx = get_any(val, ["query_x","query_images","qx","queryX"])
        qy = get_any(val, ["query_y","query_labels","qy","queryY"])
        if sx is not None and sy is not None and qx is not None and qy is not None:
            SX, SY = _coerce_xy(sx, sy)
            QX, QY = _coerce_xy(qx, qy)
            return torch.cat([SX, QX], 0), torch.cat([SY, QY], 0)

    def _try_indices_split(val: Dict[str, Any]):
        keys = set(val.keys())
        has_xy = any(k in keys for k in ("x","X","images")) and any(k in keys for k in ("y","Y","labels"))
        has_idx = any(k in keys for k in ("support_idx","support_indices")) and any(k in keys for k in ("query_idx","query_indices"))
        if not (has_xy and has_idx): return None
        x = val.get("x", val.get("X", val.get("images")))
        y = val.get("y", val.get("Y", val.get("labels")))
        sup_idx = _to_tensor(val.get("support_idx", val.get("support_indices"))).long().view(-1)
        qry_idx = _to_tensor(val.get("query_idx", val.get("query_indices"))).long().view(-1)
        X, Y = _coerce_xy(x, y)
        SX, SY = X[sup_idx], Y[sup_idx]
        QX, QY = X[qry_idx], Y[qry_idx]
        return torch.cat([SX, QX], 0), torch.cat([SY, QY], 0)

    def _try_flat_xy(val: Dict[str, Any]):
        if any(k in val for k in ("x","X","images")) and any(k in val for k in ("y","Y","labels")):
            x = val.get("x", val.get("X", val.get("images")))
            y = val.get("y", val.get("Y", val.get("labels")))
            return _coerce_xy(x, y)

    tasks: List[Dict[str, torch.Tensor]] = []

    def _add_one(val, angle_tag):
        if not isinstance(val, dict):
            if isinstance(val, (tuple, list)) and len(val) == 2:
                X, Y = _coerce_xy(val[0], val[1])
                tasks.append({"x": X, "y": Y, "angle": angle_tag})
                return True
            return False
        for try_fn in (_try_support_query_dict, _try_prefixed_keys, _try_indices_split, _try_flat_xy):
            out = try_fn(val)
            if out is not None:
                X, Y = out
                tasks.append({"x": X, "y": Y, "angle": val.get("angle", angle_tag)})
                return True
        return False

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            if not _add_one(v, i):
                raise ValueError(f"Unrecognized task structure at index {i}")
    elif isinstance(obj, dict):
        container = obj.get("tasks", obj)
        if not isinstance(container, dict):
            raise ValueError("Top-level dict had 'tasks' but it's not a dict.")
        for k, v in container.items():
            if not _add_one(v, k):
                raise ValueError(f"Unrecognized task structure for key {k}")
    else:
        raise ValueError("Unrecognized pickle top-level type (need list or dict).")

    for i, t in enumerate(tasks):
        if t["x"].dim() != 2 or t["x"].shape[1] != 784:
            raise ValueError(f"Task {i} x has shape {tuple(t['x'].shape)}, expected [N,784]")
        if t["y"].dim() != 1 or t["x"].shape[0] != t["y"].shape[0]:
            raise ValueError(f"Task {i} y shape mismatch")
    print(f"Parsed {len(tasks)} tasks from {os.path.basename(pkl_path)}")
    return tasks


def build_recurring_rotmnist_train(tasks: List[Dict[str, torch.Tensor]]) -> Tuple[List[Dict[str, torch.Tensor]], Dict[int, int]]:
    even_ids = [i for i in range(len(tasks)) if i % 2 == 0 and i <= 18]
    if len(even_ids) != 10:
        raise ValueError("Expected exactly 20 tasks (ids 0..19) so we can split even ids into 10 pairs.")

    new_tasks: List[Dict[str, torch.Tensor]] = []
    recurring_to_orig: Dict[int, int] = {}

    for orig_id in even_ids:
        task = tasks[orig_id]
        x_src = task["x"]
        y_src = task["y"]
        total = x_src.size(0)
        if total % 2 != 0:
            raise ValueError(
                f"Task {orig_id} has {total} samples; recurring split requires an even count."
            )
        mid = total // 2
        bounds = ((0, mid, 0), (mid, total, 1))
        for start, end, split_idx in bounds:
            xs = x_src[start:end].clone()
            ys = y_src[start:end].clone()
            tid = orig_id * 2 + split_idx
            recurring_to_orig[tid] = orig_id
            new_tasks.append({
                "x": xs,
                "y": ys,
                "angle": float(task.get("angle", 0.0)),
                "task_id": tid,
                "orig_task_id": orig_id,
                "split_index": split_idx,
            })

    new_tasks.sort(key=lambda t: t["task_id"])
    return new_tasks, recurring_to_orig

# ==============================
# Model: simple MLP
# ==============================
class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

    def forward_with_params(self, x, params: Dict[str, torch.Tensor]):
        x = F.relu(F.linear(x, params["fc1.weight"], params["fc1.bias"]))
        x = F.relu(F.linear(x, params["fc2.weight"], params["fc2.bias"]))
        return F.linear(x, params["head.weight"], params["head.bias"])


class ConvMetaClassifier(nn.Module):
    """4-layer ConvNet + linear head used for nonstationary benchmark episodes."""

    def __init__(self,
                 n_way: int,
                 input_channels: int = 3,
                 input_shape: Tuple[int, int] = (28, 28),
                 num_conv_layer: int = 4,
                 num_filter: int = 64,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 maxpool_kernel_size: int = 2):
        super().__init__()

        layers = OrderedDict()
        cur_in = input_channels
        for i in range(num_conv_layer):
            idx = i + 1
            layers[f"conv{idx}"] = nn.Conv2d(cur_in, num_filter,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding)
            layers[f"bn{idx}"] = nn.BatchNorm2d(num_filter, affine=True)
            layers[f"relu{idx}"] = nn.ReLU(inplace=True)
            layers[f"pool{idx}"] = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
            cur_in = num_filter

        self.encoder = nn.Sequential(layers)

        def _down_dim(sz: int) -> int:
            val = sz
            for _ in range(num_conv_layer):
                val = max(1, val // maxpool_kernel_size)
            return val

        h_out = _down_dim(int(input_shape[0]))
        w_out = _down_dim(int(input_shape[1]))
        self.output_dim = num_filter * h_out * w_out
        self.head = nn.Linear(self.output_dim, n_way)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for module in self.encoder:
            h = module(h)
        h = h.view(h.size(0), -1)
        return self.head(h)

    def _apply_module(self, x: torch.Tensor, module: nn.Module, params: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            weight = params.get(f"{prefix}.weight", module.weight)
            bias = params.get(f"{prefix}.bias", module.bias)
            return F.conv2d(x, weight, bias, module.stride, module.padding, module.dilation, module.groups)
        if isinstance(module, nn.BatchNorm2d):
            weight = params.get(f"{prefix}.weight", module.weight)
            bias = params.get(f"{prefix}.bias", module.bias)
            running_mean = module.running_mean
            running_var = module.running_var
            return F.batch_norm(x, running_mean, running_var, weight, bias, training=False, eps=module.eps)
        if isinstance(module, nn.ReLU):
            return F.relu(x, inplace=False)
        if isinstance(module, nn.MaxPool2d):
            return F.max_pool2d(x,
                                module.kernel_size,
                                module.stride,
                                module.padding,
                                module.dilation,
                                module.ceil_mode,
                                module.return_indices)
        return module(x)

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = x
        for name, module in self.encoder.named_children():
            h = self._apply_module(h, module, params, f"encoder.{name}")
        h = h.view(h.size(0), -1)
        weight = params.get("head.weight", self.head.weight)
        bias = params.get("head.bias", self.head.bias)
        return F.linear(h, weight, bias)


class VersaConvBlock(nn.Module):
    """
    VERSA-style conv block: Conv(3x3, same padding) → BatchNorm (optional) → ReLU → Dropout → MaxPool(2x2)
    Xavier normal init for conv weights to mirror TF xavier initializer.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 use_batch_norm=True, dropout_p=0.1, pool_kernel=2):
        super().__init__()
        bias = not use_batch_norm
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(out_ch, momentum=1.0, affine=True, track_running_stats=False)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.pool_kernel = pool_kernel
        nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, self.pool_kernel)
        return x


class VersaMetaClassifier(nn.Module):
    """VERSA-style 4-layer ConvNet encoder + linear head (n_way logits)."""
    def __init__(self,
                 n_way: int,
                 input_channels: int = 3,
                 input_shape: Tuple[int, int] = (28, 28),
                 num_conv_layer: int = 4,
                 num_filter: int = 64,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 maxpool_kernel_size: int = 2,
                 use_batch_norm: bool = True,
                 dropout_p: float = 0.1):
        super().__init__()
        layers = []
        cur_in = input_channels
        for _ in range(num_conv_layer):
            layers.append(VersaConvBlock(cur_in, num_filter,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         use_batch_norm=use_batch_norm,
                                         dropout_p=dropout_p,
                                         pool_kernel=maxpool_kernel_size))
            cur_in = num_filter
        self.encoder = nn.Sequential(*layers)

        def _down(sz: int, k: int, reps: int) -> int:
            v = sz
            for _ in range(reps):
                v = max(1, v // k)
            return v

        h_out = _down(int(input_shape[0]), maxpool_kernel_size, num_conv_layer)
        w_out = _down(int(input_shape[1]), maxpool_kernel_size, num_conv_layer)
        self.output_dim = num_filter * h_out * w_out
        self.head = nn.Linear(self.output_dim, n_way)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.head(h)

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = x
        # Walk encoder blocks and apply ops with provided fast weights
        for idx, block in enumerate(self.encoder):
            # Conv
            w = params.get(f"encoder.{idx}.conv.weight", block.conv.weight)
            b = params.get(f"encoder.{idx}.conv.bias", block.conv.bias)
            h = F.conv2d(h, w, b, stride=block.conv.stride, padding=block.conv.padding,
                         dilation=block.conv.dilation, groups=block.conv.groups)
            # BatchNorm (if present)
            if getattr(block, "bn", None) is not None:
                bn_w = params.get(f"encoder.{idx}.bn.weight", block.bn.weight)
                bn_b = params.get(f"encoder.{idx}.bn.bias", block.bn.bias)
                # Track_running_stats=False → no running stats; compute BN from batch
                h = F.batch_norm(h, running_mean=None, running_var=None,
                                 weight=bn_w, bias=bn_b,
                                 training=True, momentum=block.bn.momentum, eps=block.bn.eps)
            # ReLU
            h = F.relu(h, inplace=False)
            # Dropout
            if getattr(block, "dropout", None) is not None and block.dropout.p > 0:
                h = F.dropout(h, p=block.dropout.p, training=self.training)
            # MaxPool
            h = F.max_pool2d(h, kernel_size=block.pool_kernel)

        h = h.view(h.size(0), -1)
        w = params.get("head.weight", self.head.weight)
        b = params.get("head.bias", self.head.bias)
        return F.linear(h, w, b)
# ==============================
# Per-task buffers + Global cap/eviction
# ==============================
class TaskBuffer:
    """Holds samples for one task. Backed by lists so we can remove at random."""
    def __init__(self):
        self.x: List[torch.Tensor] = []
        self.y: List[int] = []

    def add_block(self, xb: torch.Tensor, yb: torch.Tensor):
        for i in range(xb.size(0)):
            self.x.append(xb[i].clone())
            self.y.append(int(yb[i]))

    def size(self): return len(self.x)

    def pop_random(self, k: int = 1) -> int:
        """Remove k random samples from this buffer. Returns removed count."""
        removed = 0
        for _ in range(min(k, self.size())):
            j = random.randrange(self.size())
            self.x.pop(j); self.y.pop(j)
            removed += 1
        return removed

    def sample_episode(self, K: int, Q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.size()
        idx = torch.randperm(n)[:(K + Q)] if n >= (K + Q) else torch.randint(0, n, (K + Q,))
        xs = torch.stack([self.x[i] for i in idx], dim=0)
        ys = torch.tensor([self.y[i] for i in idx], dtype=torch.long)
        return xs[:K], ys[:K], xs[K:], ys[K:]

class GlobalBuffer:
    """Manages a dict of TaskBuffer with a global sample cap and random eviction."""
    def __init__(self, cap: int):
        self.cap = cap
        self.buffers: Dict[int, TaskBuffer] = defaultdict(TaskBuffer)
        self.total = 0

    def ensure_space(self, need: int):
        """Randomly evict samples across tasks until there is 'need' free slots."""
        excess = self.total + need - self.cap
        while excess > 0 and self.total > 0:
            # deterministic given python RNG seed
            non_empty = [tid for tid, buf in self.buffers.items() if buf.size() > 0]
            if not non_empty: break
            tid = random.choice(non_empty)
            removed = self.buffers[tid].pop_random(1)
            self.total -= removed
            excess -= removed

    def add_block(self, tid: int, xb: torch.Tensor, yb: torch.Tensor) -> int:
        n = xb.size(0)
        self.ensure_space(n)
        self.buffers[tid].add_block(xb, yb)
        self.total += n
        return n

    def size(self, tid: int) -> int:
        return self.buffers[tid].size()

    def sample_episode(self, tid: int, K: int, Q: int):
        return self.buffers[tid].sample_episode(K, Q)

# ==============================
# Determinism helpers
# ==============================
def set_model_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def permute_tasks_inplace(tasks, data_seed: int, tag: str = "train"):
    """
    For each training task, generate a fixed permutation with a per-task generator
    derived from 'data_seed'. Stores the indices for auditing at key '{tag}_perm_idx'.
    """
    for i, t in enumerate(tasks):
        N = t["x"].size(0)
        derived = (data_seed + 17) * 1_000_003 + i * 97_003
        g = torch.Generator()
        g.manual_seed(int(derived) & 0x7FFFFFFF)
        idx = torch.randperm(N, generator=g)
        t["x"] = t["x"][idx]
        t["y"] = t["y"][idx]
        t[f"{tag}_perm_idx"] = idx  # optional for logging

# ==============================
# MAML inner loop
# ==============================
def cross_entropy(logits, y): return F.cross_entropy(logits, y)

def apply_grad_step(params: Dict[str, torch.Tensor], grads: Dict[str, torch.Tensor], lr: float):
    return {n: params[n] - lr * grads[n] for n in params}

def maml_inner_step(model: MLP,
                    params: Dict[str, torch.Tensor],
                    x_supp: torch.Tensor,
                    y_supp: torch.Tensor,
                    inner_lr: float,
                    steps: int,
                    first_order: bool) -> Dict[str, torch.Tensor]:
    cur = {k: v for k, v in params.items()}
    for _ in range(steps):
        logits = model.forward_with_params(x_supp, cur)
        loss = cross_entropy(logits, y_supp)
        grads_list = torch.autograd.grad(
            loss, list(cur.values()),
            create_graph=(not first_order), retain_graph=(not first_order)
        )
        grads = {k: g for k, g in zip(cur.keys(), grads_list)}
        cur = apply_grad_step(cur, grads, inner_lr)
    return cur

# ==============================
# Evaluation helpers — per-task accuracy with adaptation
# ==============================
def build_eval_contexts(tasks: List[Dict[str, torch.Tensor]],
                        support: int,
                        query: int,
                        seed: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Deterministically pick support/query indices for each task."""
    total = support + query
    contexts: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx, task in enumerate(tasks):
        N = task["x"].size(0)
        if N == 0:
            raise ValueError(f"Task {idx} has no samples for evaluation")
        derived = (seed + 31) * 1_000_099 + idx * 101_003
        g = torch.Generator()
        g.manual_seed(int(derived) & 0x7FFFFFFF)
        if N >= total:
            choice = torch.randperm(N, generator=g)[:total]
        else:
            choice = torch.randint(0, N, (total,), generator=g)
        contexts.append((choice[:support], choice[support:]))
    return contexts


def eval_tasks(model: MLP,
               tasks: List[Dict[str, torch.Tensor]],
               contexts: List[Tuple[torch.Tensor, torch.Tensor]],
               device,
               inner_lr: float,
               inner_steps: int,
               first_order: bool) -> List[float]:
    """Returns per-task accuracy after an inner-loop adaptation on support splits."""
    if len(tasks) != len(contexts):
        raise ValueError("tasks and contexts must have same length")

    was_training = model.training
    model.eval()
    accs: List[float] = []

    for task, (support_idx, query_idx) in zip(tasks, contexts):
        if query_idx.numel() == 0:
            raise ValueError("query size must be > 0 for evaluation")

        X = task["x"]
        Y = task["y"]
        x_supp = X[support_idx].to(device)
        y_supp = Y[support_idx].to(device)
        x_query = X[query_idx].to(device)
        y_query = Y[query_idx].to(device)

        with torch.enable_grad():
            base_params = {n: p.detach().clone().requires_grad_(True)
                           for n, p in model.named_parameters()}
            fast_params = maml_inner_step(
                model,
                base_params,
                x_supp,
                y_supp,
                inner_lr=inner_lr,
                steps=inner_steps,
                first_order=first_order
            )
        fast_params = {n: p.detach() for n, p in fast_params.items()}

        with torch.no_grad():
            logits = model.forward_with_params(x_query, fast_params)
            pred = logits.argmax(dim=1)
            accs.append((pred == y_query).float().mean().item())

    if was_training:
        model.train()
    return accs

# ==============================
# Confusion-matrix style summary over tasks
# ==============================
def confusion_matrix(result_a: torch.Tensor, log_dir: str, fname: Optional[str] = None):
    """
    result_a: (nt+1, nt) tensor of accuracies; row 0 is baseline,
              rows 1..nt are post-task rows in the same task order as columns.
    Saves a text file and a task-wise accuracy plot. Returns summary stats.
    """
    os.makedirs(log_dir, exist_ok=True)
    nt = result_a.size(1)
    assert result_a.size(0) == nt + 1, "result_a must have baseline + one row per task"

    baseline = result_a[0]            # (nt,)
    result  = result_a                # (nt+1, nt)
    square  = result[1:, :]           # (nt, nt) ignore baseline row

    # Diagonal: learned accuracy per task (after finishing that task)
    diag_vec = torch.diag(square)     # [ square[0,0], square[1,1], ..., square[nt-1, nt-1] ]

    # Final row & transfer metrics
    fin = result[-1]                  # after last task
    bwt = fin - diag_vec              # backward transfer per task

    # Forward transfer: performance on task t BEFORE seeing it (row t vs baseline col t)
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t, t] - baseline[t]

    # Additional metrics
    retained_acc = []
    learned_acc  = []
    future_acc   = []
    for t in range(1, nt+1):  # rows 1..nt
        learned = result[t, t-1].item()
        learned_acc.append(learned)

        if t > 1:
            retained = result[t, :t-1].mean().item()
        else:
            retained = 0.0
        retained_acc.append(retained)

        if t < nt:
            future = result[t, t:].mean().item()
        else:
            future = 0.0
        future_acc.append(future)

    # Save all metrics + matrix
    if fname is not None:
        path = os.path.join(log_dir, fname)
        with open(path, 'w') as f:
            print(' '.join(['%.4f' % r for r in baseline.tolist()]), file=f)
            print('|', file=f)
            for row in range(result.size(0)):
                print(' '.join(['%.4f' % r for r in result[row].tolist()]), file=f)
            print('', file=f)
            print('Diagonal Accuracy: %.4f' % diag_vec.mean().item(), file=f)
            print('Final Accuracy: %.4f' % fin.mean().item(), file=f)
            print('Backward: %.4f' % bwt.mean().item(), file=f)
            print('Forward:  %.4f' % fwt.mean().item(), file=f)
            print('\nRetained Accuracy per Task:', file=f)
            print(' '.join(['%.4f' % r for r in retained_acc]), file=f)
            print('\nLearned Accuracy per Task:', file=f)
            print(' '.join(['%.4f' % r for r in learned_acc]), file=f)
            print('\nFuture Accuracy per Task:', file=f)
            print(' '.join(['%.4f' % r for r in future_acc]), file=f)

    # Plot task-wise accuracy over time (columns)
    colors = cm.nipy_spectral(np.linspace(0, 1, result_a.size(1)))
    fig = plt.figure(figsize=(8, 8))
    data = result_a.cpu().numpy()
    for i in range(data.shape[1]):
        plt.plot(range(data.shape[0]), data[:, i], label=str(i), color=colors[i], linewidth=2)
    plt.xlabel("Evaluation checkpoint (0=baseline, then after each task)")
    plt.ylabel("Accuracy")
    plt.title("Task-wise accuracy over time")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'task_wise_accuracy.png'))
    plt.close(fig)

    # Sanity prints
    print("mean diag          :", diag_vec.float().mean().item())
    print("mean learned (diag):", float(np.mean(learned_acc)))
    print("mean retained      :", float(np.mean(retained_acc)))
    print("mean future        :", float(np.mean(future_acc)))
    print("mean recurring     :", fin.mean().item())

    stats = [
        diag_vec.mean().item(),
        fin.mean().item(),
        bwt.mean().item(),
        fwt.mean().item(),
        float(np.mean(retained_acc)),
        float(np.mean(learned_acc)),
        float(np.mean(future_acc)),
    ]
    return stats


def evaluate_nonstationary_datasets(model: nn.Module,
                                    eval_registry: Dict[str, Any],
                                    dataset_names: List[str],
                                    device,
                                    n_way: int,
                                    k_shot: int,
                                    q_query: int,
                                    eval_tasks: int,
                                    inner_lr: float,
                                    inner_steps: int,
                                    first_order: bool) -> Dict[str, Tuple[float, float]]:
    if len(dataset_names) == 0:
        return {}

    was_training = model.training
    model.eval()

    results: Dict[str, Tuple[float, float]] = {}

    for name in dataset_names:
        dataset = eval_registry[name]
        accs: List[float] = []
        for _ in range(eval_tasks):
            Sx, Sy, Qx, Qy, _, _ = dataset.sample_n_way_k_shot(n_way, k_shot, q_query)
            Sx = Sx.to(device)
            Sy = Sy.to(device)
            Qx = Qx.to(device)
            Qy = Qy.to(device)

            with torch.enable_grad():
                base_params = {n: p.detach().clone().requires_grad_(True)
                               for n, p in model.named_parameters()}
                fast_params = maml_inner_step(
                    model,
                    base_params,
                    Sx,
                    Sy,
                    inner_lr=inner_lr,
                    steps=inner_steps,
                    first_order=first_order,
                )

            fast_params = {n: p.detach() for n, p in fast_params.items()}
            with torch.no_grad():
                logits = model.forward_with_params(Qx, fast_params)
                pred = logits.argmax(dim=1)
                acc = (pred == Qy).float().mean().item()
                accs.append(acc)

        if accs:
            mean = float(np.mean(accs))
            std = float(np.std(accs))
        else:
            mean, std = 0.0, 0.0
        results[name] = (mean, std)

    if was_training:
        model.train()
    return results


def train_nonstationary(args):
    if get_dataset_registry is None:
        raise ImportError("non_stationary_datasets.py not found; nonstationary benchmark unavailable.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Enforce global buffer cap target specified for this setting (for logging parity).
    args.global_buffer_cap = 12800

    print(f"[SEED] data_seed={args.data_seed}  model_seed={args.model_seed}")
    set_model_seed(args.model_seed)

    os.makedirs(args.log_dir, exist_ok=True)

    train_registry, dataset_order = get_dataset_registry(args.data_dir, split="train")
    test_registry, _ = get_dataset_registry(args.data_dir, split=getattr(args, "nonstat_test_split", "test"))

    # Optional filtering to a subset of datasets (e.g., only 'vggflowers')
    if getattr(args, "datasets", None):
        wanted = [s.strip() for s in args.datasets.split(",") if s.strip()]
        available = set(train_registry.keys())
        missing = [w for w in wanted if w not in available]
        if missing:
            raise ValueError(f"Requested datasets not found in registry: {missing}")
        dataset_order = [n for n in dataset_order if n in set(wanted)]
        if not dataset_order:
            raise ValueError("After filtering, dataset_order is empty.")

    if len(train_registry) == 0:
        raise ValueError("No datasets found for nonstationary benchmark. Ensure prepared .pt files exist.")

    first_dataset = train_registry[dataset_order[0]]
    if not hasattr(first_dataset, "xs"):
        raise ValueError("Dataset objects must expose 'xs' tensors for shape inference.")

    sample_shape = first_dataset.xs.shape
    if len(sample_shape) < 4:
        raise ValueError("Expected dataset tensors with shape [N, C, H, W].")
    input_channels = int(sample_shape[1])
    input_shape = (int(sample_shape[2]), int(sample_shape[3]))

    if getattr(args, "encoder", "conv").lower() == "versa":
        model = VersaMetaClassifier(
            n_way=args.n_way,
            input_channels=input_channels,
            input_shape=input_shape,
            num_conv_layer=args.conv_layers,
            num_filter=args.conv_filters,
            kernel_size=args.conv_kernel,
            stride=1,
            padding=args.conv_padding,
            maxpool_kernel_size=args.conv_pool,
        ).to(device)
    else:
        model = ConvMetaClassifier(
            n_way=args.n_way,
            input_channels=input_channels,
            input_shape=input_shape,
            num_conv_layer=args.conv_layers,
            num_filter=args.conv_filters,
            kernel_size=args.conv_kernel,
            stride=1,
            padding=args.conv_padding,
            maxpool_kernel_size=args.conv_pool,
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    metrics_records: List[Dict[str, Any]] = []
    metrics_path = os.path.join(args.log_dir, "nonstationary_metrics.json")

    total_stages = len(dataset_order)
    global_updates = 0
    meta_batch = max(1, int(args.meta_batch))

    train_query_acc_sum = 0.0
    train_query_acc_count = 0
    train_support_acc_sum = 0.0
    train_support_acc_count = 0

    def run_meta_update(batch_eps: List[Tuple[torch.Tensor, ...]], stage_name: str, epoch_idx: int):
        nonlocal global_updates
        nonlocal train_query_acc_sum, train_query_acc_count
        nonlocal train_support_acc_sum, train_support_acc_count
        if not batch_eps:
            return
        opt.zero_grad()
        losses_q: List[torch.Tensor] = []
        accs_q: List[torch.Tensor] = []
        accs_s: List[torch.Tensor] = []
        for episode in batch_eps:
            Sx, Sy, Qx, Qy = episode[:4]
            Sx = Sx.to(device)
            Sy = Sy.to(device)
            Qx = Qx.to(device)
            Qy = Qy.to(device)
            base_params = {n: p for n, p in model.named_parameters()}
            fast_params = maml_inner_step(
                model,
                base_params,
                Sx,
                Sy,
                inner_lr=args.inner_lr,
                steps=args.inner_steps,
                first_order=args.first_order,
            )
            logits_q = model.forward_with_params(Qx, fast_params)
            losses_q.append(cross_entropy(logits_q, Qy))
            with torch.no_grad():
                preds = logits_q.argmax(dim=1)
                accs_q.append((preds == Qy).float().mean())
                logits_s = model.forward_with_params(Sx, fast_params)
                preds_s = logits_s.argmax(dim=1)
                accs_s.append((preds_s == Sy).float().mean())

        meta_loss = torch.stack(losses_q).mean()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        global_updates += 1

        mean_query_acc = float(torch.stack(accs_q).mean().item()) if accs_q else 0.0
        mean_support_acc = float(torch.stack(accs_s).mean().item()) if accs_s else 0.0

        train_query_acc_sum += mean_query_acc
        train_query_acc_count += 1
        train_support_acc_sum += mean_support_acc
        train_support_acc_count += 1

        if args.print_interval > 0 and (global_updates % args.print_interval) == 0:
            print(f"[UPDATE] stage={stage_name} epoch={epoch_idx+1:03d}/{args.epochs_per_dataset} "
                  f"step={global_updates:06d} meta_loss={meta_loss.item():.4f} "
                  f"context_acc={mean_support_acc*100:.2f} query_acc={mean_query_acc*100:.2f}")

    model.train()

    for stage_idx, dataset_name in enumerate(dataset_order):
        dataset = train_registry[dataset_name]
        print(f"\n[STAGE] {stage_idx+1}/{total_stages} dataset={dataset_name}")

        for epoch in range(args.epochs_per_dataset):
            batch_eps: List[Tuple[torch.Tensor, ...]] = []
            for _ in range(args.episodes_per_epoch):
                episode = dataset.sample_n_way_k_shot(args.n_way, args.k_shot, args.q_query)
                batch_eps.append(episode)
                if len(batch_eps) == meta_batch:
                    run_meta_update(batch_eps, dataset_name, epoch)
                    batch_eps = []
            if batch_eps:
                run_meta_update(batch_eps, dataset_name, epoch)

        print(f"[BOUNDARY] Completed dataset '{dataset_name}' ({stage_idx+1}/{total_stages}).")
        if train_query_acc_count > 0:
            mean_q = (train_query_acc_sum / train_query_acc_count) * 100.0
            print(f"[TRAIN] running query accuracy across updates: {mean_q:.2f}%")
        if train_support_acc_count > 0:
            mean_s = (train_support_acc_sum / train_support_acc_count) * 100.0
            print(f"[TRAIN] running context accuracy across updates: {mean_s:.2f}%")
        train_query_acc_sum = 0.0
        train_query_acc_count = 0
        train_support_acc_sum = 0.0
        train_support_acc_count = 0

        eval_names = dataset_order[:stage_idx + 1]
        eval_results = evaluate_nonstationary_datasets(
            model,
            test_registry,
            eval_names,
            device,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            eval_tasks=args.eval_tasks,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            first_order=args.first_order,
        )

        stage_avg = float(np.mean([eval_results[n][0] for n in eval_names])) if eval_names else 0.0
        print("[EVAL]")
        for name in eval_names:
            mean, std = eval_results[name]
            print(f"    {name:>12}: {mean*100:.2f} ± {std*100:.2f}")
        print(f"    {'Average':>12}: {stage_avg*100:.2f}")

        metrics_records.append({
            "stage": stage_idx + 1,
            "dataset": dataset_name,
            "results": {name: {"mean": eval_results[name][0], "std": eval_results[name][1]} for name in eval_names},
            "average": stage_avg,
        })

        with open(metrics_path, "w") as f:
            json.dump(metrics_records, f, indent=2)

    print("\n[NONSTATIONARY] Training complete. Metrics written to", metrics_path)


# ==============================
# Training loop — Algorithm 1 style (fixed order, one pass)
# ==============================
def train(args):
    if getattr(args, "nonstationary_benchmark", False):
        train_nonstationary(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print(f"[SEED] data_seed={args.data_seed}  model_seed={args.model_seed}")
    set_model_seed(args.model_seed)

    # dataset
    if args.pt_file:
        pt_path = args.pt_file if os.path.isabs(args.pt_file) else os.path.join(args.data_dir, args.pt_file)
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Could not find {pt_path}")
        train_tasks, test_tasks = load_tasks_pt(pt_path)
    else:
        train_tasks = load_tasks_pkl(os.path.join(args.data_dir, args.train_pkl))
        test_tasks  = load_tasks_pkl(os.path.join(args.data_dir, args.test_pkl))

    use_recurring = args.recurring_rot_mnist or args.recurring_perm_mnist
    if args.recurring_rot_mnist and args.recurring_perm_mnist:
        raise ValueError("Use only one of --recurring_rot_mnist or --recurring_perm_mnist.")
    recurring_to_orig: Dict[int, int] = {}
    if use_recurring:
        train_tasks, recurring_to_orig = build_recurring_rotmnist_train(train_tasks)
        print(f"[RECURRING] expanded to {len(train_tasks)} recurring train tasks from even task ids.")

    def orig_id(tid: int) -> int:
        return recurring_to_orig.get(tid, tid)

    # fixed per-task permutations for TRAIN tasks
    permute_tasks_inplace(train_tasks, args.data_seed, tag="train")
    os.makedirs(args.log_dir, exist_ok=True)
    try:
        torch.save([t["train_perm_idx"] for t in train_tasks],
                   os.path.join(args.log_dir, f"perms_data{args.data_seed}.pt"))
    except Exception:
        pass
    print("[PERM] first-5 indices of first 3 train tasks:",
          [t["train_perm_idx"][:5].tolist() for t in train_tasks[:3]])

    print(f"Loaded {len(train_tasks)} training tasks.")

    # task order (one pass, exactly once)
    pt_name = os.path.basename(args.pt_file) if args.pt_file else ""
    is_perm = "permutation" in pt_name.lower()
    if args.task_order:
        order = [int(x) for x in args.task_order.split(",")]
    elif use_recurring and is_perm:
        order = sorted(recurring_to_orig.keys())
    elif use_recurring and not is_perm:
        order = [8, 28, 9, 29, 12, 32, 13, 33, 16, 36, 17, 37, 0, 20, 1, 21, 4, 24, 5, 25]
    elif is_perm:
        order = list(range(len(train_tasks)))
    else:
        order = [18, 1, 19, 8, 10, 17, 6, 13, 4, 2, 5, 14, 9, 7, 16, 11, 3, 0, 15, 12]
    assert len(order) == len(train_tasks), "task_order length must equal num tasks"

    # reorder test tasks to column alignment with 'order'
    if use_recurring:
        missing = [tid for tid in order if tid not in recurring_to_orig]
        if missing:
            raise ValueError(f"Order contains task ids not built by recurring loader: {missing}")
        test_tasks_ordered = [test_tasks[recurring_to_orig[tid]] for tid in order]
    else:
        test_tasks_ordered = [test_tasks[i] for i in order]
    nt = len(order)

    eval_contexts = build_eval_contexts(
        test_tasks_ordered,
        support=args.support,
        query=args.query,
        seed=args.data_seed
    )

    # state
    if use_recurring:
        train_task_by_id = {task["task_id"]: task for task in train_tasks}
        per_task_ptr: Dict[int, int] = {tid: 0 for tid in train_task_by_id.keys()}
    else:
        train_task_by_id = {}
        per_task_ptr = [0] * len(train_tasks)
    buffers = GlobalBuffer(cap=args.global_buffer_cap)  # global cap with random eviction
    model = MLP(in_dim=784, hidden=args.hidden, num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    take_per_iter = args.support + args.query
    global_updates = 0

    def task_exhausted(tid: int) -> bool:
        task = train_task_by_id[tid] if use_recurring else train_tasks[tid]
        return per_task_ptr[tid] >= task["x"].size(0)

    def ingest_from_task(task_id: int, take: int) -> int:
        task = train_task_by_id[task_id] if use_recurring else train_tasks[task_id]
        x, y = task["x"], task["y"]
        start, end = per_task_ptr[task_id], min(per_task_ptr[task_id] + take, x.size(0))
        if end <= start: return 0
        added = end - start
        buffers.add_block(task_id, x[start:end], y[start:end])  # enforces global cap internally
        per_task_ptr[task_id] = end
        return added

    # ---- Evaluation logging containers (baseline + after each task) ----
    all_rows: List[torch.Tensor] = []

    # Baseline row (untrained model)
    row0 = eval_tasks(
        model,
        test_tasks_ordered,
        eval_contexts,
        device,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        first_order=args.first_order,
    )
    all_rows.append(torch.tensor(row0, dtype=torch.float32))

    # Training — one pass over tasks in the given order
    model.train()
    for t_idx, t in enumerate(order):                   # go through each task once
        task = train_task_by_id[t] if use_recurring else train_tasks[t]
        angle = task["angle"]

        while not task_exhausted(t):                    # keep ingesting this task until exhausted
            _ = ingest_from_task(t, take_per_iter)

            # build eligible set from tasks we've seen so far in this order (preserve order; no Python set)
            seen_prefix = order[:order.index(t) + 1]
            eligible = [tid for tid in seen_prefix if buffers.size(tid) >= (args.support + args.query)]
            if len(eligible) == 0:
                continue

            chosen = eligible if len(eligible) < args.meta_batch else random.sample(eligible, args.meta_batch)

            # episodes
            episodes = []
            for tid in chosen:
                xs, ys, xq, yq = buffers.sample_episode(tid, args.support, args.query)
                episodes.append((xs.to(device), ys.to(device), xq.to(device), yq.to(device)))

            # meta-update (MAML)
            opt.zero_grad()
            base_params = {n: p for n, p in model.named_parameters()}  # real params (no detach)

            losses_q = []
            for (xs, ys, xq, yq) in episodes:
                fast_params = maml_inner_step(
                    model, base_params, xs, ys,
                    inner_lr=args.inner_lr, steps=args.inner_steps, first_order=args.first_order
                )
                logits_q = model.forward_with_params(xq, fast_params)
                loss_q = cross_entropy(logits_q, yq)
                losses_q.append(loss_q)

            meta_loss = torch.stack(losses_q).mean()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            global_updates += 1

            if global_updates % args.print_interval == 0:
                seen_so_far = per_task_ptr[t]
                total_t = task["x"].size(0)
                pct = 100.0 * seen_so_far / total_t
                print(f"Step {global_updates:6d} | task={t:02d} angle={angle:6.2f} "
                      f"| progress={pct:5.1f}% | meta_loss={meta_loss.item():.4f} "
                      f"| global_buf={buffers.total}")

            if args.iters > 0 and global_updates >= args.iters:
                print("Reached --iters cap; stopping.")
                break

        if args.iters > 0 and global_updates >= args.iters:
            break

        # ======= evaluation row AFTER finishing task t =======
        row = eval_tasks(
            model,
            test_tasks_ordered,
            eval_contexts,
            device,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            first_order=args.first_order,
        )
        row_t = torch.tensor(row, dtype=torch.float32)
        all_rows.append(row_t)  # append post-task row

        # --- DEBUG PRINTS AFTER EACH TASK ---
        pos = t_idx                        # column position in 'order'
        learned  = row_t[pos].item()
        retained = row_t[:pos].mean().item() if pos > 0 else 0.0
        future   = row_t[pos+1:].mean().item() if pos < (nt - 1) else 0.0

        print(f"[AFTER TASK idx={t_idx:02d} (task_id={t:02d}, angle={angle:6.2f})] "
              f"learned(diag)={learned:.4f}  retained={retained:.4f}  future={future:.4f}")

        print(f"    current_orig = {orig_id(t)}")
        if t_idx > 0:
            retained_orig = [orig_id(tid) for tid in order[:t_idx]]
            print(f"    retained_orig_tasks = {retained_orig}")
        else:
            print("    retained_orig_tasks = []")
        if t_idx < nt - 1:
            future_orig = [orig_id(tid) for tid in order[t_idx+1:]]
            print(f"    future_orig_tasks   = {future_orig}")
        else:
            print("    future_orig_tasks   = []")

        if len(all_rows) >= 2:
            diag_so_far = [ all_rows[i][i-1].item() for i in range(1, len(all_rows)) ]
            print("    diag_so_far =", ["%.4f" % v for v in diag_so_far])
        print("    row =", ["%.4f" % v for v in row_t.tolist()])

    # Save model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
        print(f"Saved model to {args.save_path}")

    # ======= build matrix + confusion_matrix metrics =======
    result_a = torch.stack(all_rows, dim=0)  # shape: (nt+1, nt)
    stats = confusion_matrix(result_a=result_a, log_dir=args.log_dir, fname="ftml_eval.txt")
    print("[EVAL] Summary stats (diag_acc, final_acc, bwt, fwt, retained_mean, learned_mean, future_mean):")
    print([float(f"{s:.4f}") for s in stats])

# ==============================
# CLI
# ==============================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--pt_file", type=str, default="", help="If set, use this .pt file (train/test).")
    p.add_argument("--train_pkl", type=str, default="train.pkl")
    p.add_argument("--test_pkl", type=str, default="test.pkl")
    p.add_argument("--recurring_rot_mnist", action="store_true",
                   help="Use recurring RotMNIST setting (even ids split into halves).")
    p.add_argument("--recurring_perm_mnist", action="store_true",
                   help="Use recurring PermMNIST setting (even ids split into halves).")
    p.add_argument("--datasets", type=str, default="",
                   help="Comma-separated subset of datasets to train/eval in nonstationary mode (e.g., 'vggflowers').")

    # model / MAML
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--inner_steps", type=int, default=1)
    p.add_argument("--inner_lr", type=float, default=0.01)
    p.add_argument("--meta_lr", type=float, default=1e-3)
    p.add_argument("--first_order", action="store_true")

    # FTML stream & buffers
    p.add_argument("--support", type=int, default=10)
    p.add_argument("--query", type=int, default=100)
    p.add_argument("--meta_batch", type=int, default=2)

    # global buffer cap (samples across all tasks)
    p.add_argument("--global_buffer_cap", type=int, default=30000)

    p.add_argument("--iters", type=int, default=0, help="Optional max meta-updates; 0 = no cap.")
    p.add_argument("--print_interval", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save_path", type=str, default="checkpoints/ftml_mlp.pt")

    # evaluation / logging
    p.add_argument("--eval_batch", type=int, default=4096)
    p.add_argument("--log_dir", type=str, default="logs")

    # fixed one-pass order (comma separated list of task ids)
    p.add_argument("--task_order", type=str, default="")

    # Nonstationary benchmark options (ignored unless --nonstationary_benchmark is set)
    p.add_argument("--nonstationary_benchmark", action="store_true",
                   help="Use continual multi-dataset benchmark instead of RotMNIST stream.")
    p.add_argument("--epochs_per_dataset", type=int, default=50,
                   help="Epochs per dataset in nonstationary benchmark.")
    p.add_argument("--episodes_per_epoch", type=int, default=32,
                   help="Episodes sampled per epoch in nonstationary benchmark.")
    p.add_argument("--n_way", type=int, default=5,
                   help="Classes per episode in nonstationary benchmark.")
    p.add_argument("--k_shot", type=int, default=5,
                   help="Support shots per class in nonstationary benchmark.")
    p.add_argument("--q_query", type=int, default=15,
                   help="Query samples per class in nonstationary benchmark.")
    p.add_argument("--eval_tasks", type=int, default=100,
                   help="Evaluation episodes per dataset boundary in nonstationary benchmark.")
    p.add_argument("--conv_filters", type=int, default=64,
                   help="Conv filters for nonstationary encoder.")
    p.add_argument("--conv_layers", type=int, default=4,
                   help="Number of conv layers for nonstationary encoder.")
    p.add_argument("--conv_kernel", type=int, default=3,
                   help="Kernel size for conv layers in nonstationary encoder.")
    p.add_argument("--conv_padding", type=int, default=1,
                   help="Padding for conv layers in nonstationary encoder.")
    p.add_argument("--conv_pool", type=int, default=2,
                   help="Max-pool kernel size per conv block in nonstationary encoder.")
    p.add_argument("--encoder", type=str, default="conv",
                   help="Encoder backbone for nonstationary mode: 'conv' (default) or 'versa'.")

    # seeds: separate data vs model/training RNG
    p.add_argument("--data_seed",  type=int, default=0,   help="Seed for per-task permutations (data order)")
    p.add_argument("--model_seed", type=int, default=0,   help="Seed for model init & training-time RNG")
    # deprecated combined seed (kept for convenience)
    p.add_argument("--seed", type=int, default=None, help="Deprecated; if set, overrides both seeds")

    args = p.parse_args()
    if args.seed is not None:
        args.data_seed = args.seed
        args.model_seed = args.seed
    train(args)

if __name__ == "__main__":
    main()
