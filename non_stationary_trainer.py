import os
import json
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Dict, Tuple, List, Any
import random

from non_stationary_datasets import get_dataset_registry

import collections

def _safe_model_state_export(model: torch.nn.Module) -> dict:
    """
    Export model weights/buffers without calling Module.state_dict(),
    to avoid modules that override it with incompatible signatures.
    Returns a flat dict name->tensor on CPU.
    """
    export = collections.OrderedDict()
    # Parameters
    for name, param in model.named_parameters(recurse=True):
        if param is None:
            continue
        export[name] = param.detach().cpu()
    # Buffers
    for name, buf in model.named_buffers(recurse=True):
        if buf is None:
            continue
        # Avoid overwriting params with same name (rare, but guard)
        if name not in export:
            export[name] = buf.detach().cpu()
    return export

def _safe_model_state_load(model: torch.nn.Module, state: dict, strict: bool = False):
    """
    Load tensors into model params/buffers by name and shape.
    Does not invoke Module.load_state_dict() to avoid incompatible overrides.
    """
    with torch.no_grad():
        # Build mapping of current model tensors
        cur = {}
        for n, p in model.named_parameters(recurse=True):
            cur[n] = p
        for n, b in model.named_buffers(recurse=True):
            if n not in cur:
                cur[n] = b
        missing = []
        mismatched = []
        for k, v in state.items():
            if k in cur:
                t = cur[k]
                if t.shape == v.shape:
                    t.copy_(v.to(t.device))
                else:
                    mismatched.append((k, tuple(t.shape), tuple(v.shape)))
            else:
                missing.append(k)
        if strict:
            if missing or mismatched:
                msg = []
                if missing:
                    msg.append(f"Missing keys not found in model: {missing[:8]}{' ...' if len(missing)>8 else ''}")
                if mismatched:
                    msg.append(f"Shape mismatch: {len(mismatched)} tensors (e.g., {mismatched[:3]})")
                raise RuntimeError("Strict load failed: " + " | ".join(msg))


class EpisodicBatcher(IterableDataset):
    def __init__(self, dataset, n_way: int, k_shot: int, q_query: int, num_tasks: int):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks

    def __iter__(self):
        for _ in range(self.num_tasks):
            yield self.dataset.sample_n_way_k_shot(self.n_way, self.k_shot, self.q_query)

    def __len__(self):
        return self.num_tasks


class ContinualMetaTrainer:
    def __init__(
        self,
        model,
        # memory,
        args,
        data_root: str,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        epochs_per_dataset: int = 50,
        device: str = "cuda",
        known_boundary: bool = True,
        log_path: str = None,
    ):
        self.model = model
        # self.model.memory = memory
        self.args = args
        self.data_root = data_root

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.epochs_per_dataset = epochs_per_dataset
        self.device = device
        self.known_boundary = known_boundary

        # Train registry and fixed order
        self.registry, self.order = get_dataset_registry(self.data_root, split="train")
        self.global_task_counter = 0  # boundary id for memory hooks

        # Build evaluation registries
        self.val_split_name = getattr(self.args, "val_split_name", "val")
        self.test_split_name = getattr(self.args, "test_split_name", "test")
        self.eval_registry_val, _ = get_dataset_registry(self.data_root, split=self.val_split_name)
        self.eval_registry_test, _ = get_dataset_registry(self.data_root, split=self.test_split_name)

        # Lower-triangular accuracy matrix over dataset boundaries (test accuracies)
        self.test_acc_matrix = [[None for _ in range(len(self.order))] for __ in range(len(self.order))]

        # Logging
        log_dir = getattr(self.args, "log_dir", ".")
        self.log_path = log_path or os.path.join(log_dir, "continual_eval_log.txt")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write("=== Continual Meta-Training Session Start ===\n")

        # Checkpoint dir and policy
        self.ckpt_dir = getattr(self.args, "ckpt_dir", os.path.join(log_dir, "checkpoints"))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_every_epochs = int(getattr(self.args, "ckpt_every_epochs", 5))  # 0 to disable
        self.keep_last_k = int(getattr(self.args, "keep_last_k", 3))

        self.model.to(device)

    # ---------------------- Evaluation helpers (unchanged) ----------------------
    # def _episode_to_batch(self, Sx, Sy, Qx, Qy):
    #     Qx = Qx.to(self.device)
    #     Qy = Qy.to(self.device)
    #     Sx = Sx.to(self.device)
    #     Sy = Sy.to(self.device)

    #     Bq = Qx.size(0)
    #     K = self.k_shot
    #     uniq = torch.unique(Sy)
    #     Sx_by_class = {int(c.item()): Sx[Sy == c] for c in uniq}
    #     domain_images = []
    #     domain_labels = []
    #     for i in range(Bq):
    #         c = int(Qy[i].item())
    #         pool = Sx_by_class[c]
    #         if pool.size(0) >= K:
    #             idx = torch.randperm(pool.size(0), device=self.device)[:K]
    #         else:
    #             idx = torch.randint(low=0, high=pool.size(0), size=(K,), device=self.device)
    #         dom_i = pool.index_select(0, idx)
    #         lab_i = torch.full((K,), c, dtype=torch.long, device=self.device)
    #         domain_images.append(dom_i)
    #         domain_labels.append(lab_i)
    #     domain_images = torch.stack(domain_images, dim=0)
    #     domain_labels = torch.stack(domain_labels, dim=0)
    #     return Qx, Qy, domain_images, domain_labels

    def _episode_to_batch(self, Sx, Sy, Qx, Qy):
        """
        Build per-episode batch where the same context (support) set is repeated
        for every query sample. No per-query sampling from the support pool.

        Inputs:
        - Sx: [Ns, C, H, W] support images for the episode (Ns = n_way * k_shot)
        - Sy: [Ns] support labels in episodic (local) class ids 0..n_way-1 (or global if the sampler does so)
        - Qx: [B, C, H, W] query images
        - Qy: [B] query labels (consistent with Sy’s id space)

        Returns:
        - Qx, Qy unchanged on device
        - domain_images: [B, Kc, C, H, W] same support set repeated for each query
        - domain_labels: [B, Kc] labels repeated likewise
        """
        device = self.device
        Qx = Qx.to(device)
        Qy = Qy.to(device)
        Sx = Sx.to(device)
        Sy = Sy.to(device)

        B = Qx.size(0)

        # Decide context size Kc
        # If args.context_size is provided, use it; else use the full support set
        Kc_target = int(getattr(self.args, "context_size", Sx.size(0)))
        Kc_target = max(1, Kc_target)

        # If we have fewer support samples than Kc_target, we can either:
        #  - pad by random sampling with replacement, or
        #  - just repeat cyclically.
        # Below: sample with replacement to exactly reach Kc_target; if more, truncate.
        Ns = Sx.size(0)
        if Ns == Kc_target:
            Sx_ctx = Sx
            Sy_ctx = Sy
        elif Ns > Kc_target:
            idx = torch.randperm(Ns, device=device)[:Kc_target]
            Sx_ctx = Sx.index_select(0, idx)
            Sy_ctx = Sy.index_select(0, idx)
        else:
            # Ns < Kc_target: sample with replacement
            idx = torch.randint(low=0, high=Ns, size=(Kc_target,), device=device)
            Sx_ctx = Sx.index_select(0, idx)
            Sy_ctx = Sy.index_select(0, idx)

        # Repeat along batch (query) axis
        # Sx_ctx: [Kc, C, H, W] -> [B, Kc, C, H, W]
        domain_images = Sx_ctx.unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()
        domain_labels = Sy_ctx.unsqueeze(0).expand(B, -1).contiguous()

        return Qx, Qy, domain_images, domain_labels

    @staticmethod
    def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()

    def _forward_query_with_support(self, Sx, Sy, Qx):
        if hasattr(self.model, "predict_episode"):
            return self.model.predict_episode(Sx.to(self.device), Sy.to(self.device), Qx.to(self.device))
        if hasattr(self.model, "forward_episode"):
            return self.model.forward_episode(Sx.to(self.device), Sy.to(self.device), Qx.to(self.device))
        return self.model(Qx.to(self.device))

    def evaluate_dataset(self, ds_eval, num_eval_tasks: int = 100) -> Tuple[float, float]:
        self.model.eval()
        losses, accs = [], []
        with torch.no_grad():
            for _ in range(num_eval_tasks):
                Sx, Sy, Qx, Qy, Sg, Qg = ds_eval.sample_n_way_k_shot(self.n_way, self.k_shot, self.q_query)
                if hasattr(self.model, "eval_step"):
                    Qx_b, Qy_b, dom_x, dom_y = self._episode_to_batch(Sx, Sy, Qx, Qy)
                    loss, logits = self.model.eval_step(Qx_b, Qy_b, dom_x, dom_y)
                else:
                    logits = self._forward_query_with_support(Sx, Sy, Qx)
                    loss = torch.nn.functional.cross_entropy(logits, Qy.to(self.device))
                acc = self._accuracy_from_logits(logits, Qy.to(self.device))
                losses.append(float(loss))
                accs.append(acc)
        self.model.train()
        return float(sum(losses) / max(1, len(losses))), float(sum(accs) / max(1, len(accs)))

    def _evaluate_validation_after_epoch(self, ds_name: str, num_eval_tasks: int):
        ds_eval = self.eval_registry_val.get(ds_name, None)
        if ds_eval is None:
            print(f"[{ds_name}] eval split '{self.val_split_name}' not found; skipping validation.")
            return None, None
        val_loss, val_acc = self.evaluate_dataset(ds_eval, num_eval_tasks=num_eval_tasks)
        print(f"[{ds_name}] validation after epoch: loss={val_loss:.4f} acc={val_acc*100:.2f}%")
        with open(self.log_path, "a") as f:
            f.write(f"[VAL] {ds_name} loss={val_loss:.4f} acc={val_acc*100:.2f}%\n")
        return val_loss, val_acc

    def _evaluate_test_lower_triangle(self, upto_idx: int, num_eval_tasks: int):
        row_idx = upto_idx
        for col_idx in range(upto_idx + 1):
            ds_name = self.order[col_idx]
            ds_test = self.eval_registry_test.get(ds_name, None)
            if ds_test is None:
                print(f"[TEST] split '{self.test_split_name}' for {ds_name} not found; skipping.")
                acc = None
            else:
                _, acc = self.evaluate_dataset(ds_test, num_eval_tasks=num_eval_tasks)
                print(f"[TEST] boundary {row_idx}, dataset {ds_name}: acc={acc*100:.2f}%")
            self.test_acc_matrix[row_idx][col_idx] = acc

        with open(self.log_path, "a") as f:
            f.write(f"[LOWER_TRIANGLE] after boundary {row_idx} (0-based; inclusive):\n")
            for r in range(row_idx + 1):
                row_vals = []
                for c in range(len(self.order)):
                    v = self.test_acc_matrix[r][c]
                    row_vals.append("--" if v is None else f"{v*100:.2f}")
                f.write("  " + "\t".join(row_vals) + "\n")

    def _final_report_RA(self):
        if len(self.order) == 0:
            return
        last_row_idx = len(self.order) - 1
        last_row = self.test_acc_matrix[last_row_idx]
        vals = [v for v in last_row[:last_row_idx + 1] if v is not None]
        ra = None if len(vals) == 0 else sum(vals) / len(vals)
        with open(self.log_path, "a") as f:
            f.write("[FINAL]\n")
            if ra is None:
                f.write("RA: N/A (no test accuracies recorded)\n")
            else:
                f.write(f"RA (mean of last test accuracies over all datasets): {ra*100:.2f}%\n")
            f.write("=== Continual Meta-Training Session End ===\n")
        if ra is not None:
            print(f"[FINAL] Retained Accuracy (RA): {ra*100:.2f}%")

    # ---------------------- Checkpointing ----------------------
    def _ckpt_path(self, tag: str) -> str:
        return os.path.join(self.ckpt_dir, f"ckpt_{tag}.pt")

    def _index_path(self) -> str:
        return os.path.join(self.ckpt_dir, "index.json")

    def save_checkpoint(self, dsi: int, epoch: int, tag: str = None):
        """
        Saves model, optimizer, scheduler, memory, and trainer state.
        dsi: dataset index (0-based in self.order)
        epoch: epoch index within current dataset (0-based)
        """
        tag = tag or f"ds{dsi}_ep{epoch}"
        path = self._ckpt_path(tag)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Model-related state
        model_state = _safe_model_state_export(self.model)
        opt_state = getattr(self.model, "opt", None).state_dict() if getattr(self.model, "opt", None) else None
        sched_state = getattr(self.model, "lr_scheduler", None).state_dict() if getattr(self.model, "lr_scheduler", None) else None

        # Memory (reservoir + GMM)
        mem_state = self.model.memory.state_dict() if hasattr(self.model.memory, "state_dict") else None

        # Trainer state
        trainer_state = {
            "dsi": dsi,
            "epoch": epoch,
            "global_task_counter": self.global_task_counter,
            "test_acc_matrix": self.test_acc_matrix,
            "log_path": self.log_path,
        }

        torch.save({
            "model_state": model_state,
            "opt_state": opt_state,
            "sched_state": sched_state,
            "memory_state": mem_state,
            "trainer_state": trainer_state,
        }, path)
        print(f"[CKPT] Saved checkpoint to {path}")

        # Maintain simple index for pruning old checkpoints
        idx_path = self._index_path()
        try:
            if os.path.exists(idx_path):
                with open(idx_path, "r") as f:
                    idx = json.load(f)
            else:
                idx = {"tags": []}
        except Exception:
            idx = {"tags": []}
        idx["tags"].append(tag)
        # prune
        if self.keep_last_k > 0 and len(idx["tags"]) > self.keep_last_k:
            to_delete = idx["tags"][:-self.keep_last_k]
            for t in to_delete:
                p = self._ckpt_path(t)
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass
            idx["tags"] = idx["tags"][-self.keep_last_k:]
        with open(idx_path, "w") as f:
            json.dump(idx, f)

    def load_checkpoint(self, tag: str = None, path: str = None):
        """
        Load a checkpoint by tag or path. Restores model, optimizer/scheduler,
        memory, trainer progress, and test_acc_matrix.
        Returns (start_dsi, start_epoch) as the resume point.
        """
        if path is None:
            if tag is None:
                # default to latest
                idx_path = self._index_path()
                if os.path.exists(idx_path):
                    with open(idx_path, "r") as f:
                        idx = json.load(f)
                    if idx.get("tags"):
                        tag = idx["tags"][-1]
            if tag is None:
                raise FileNotFoundError("No checkpoint tag provided and no index.json found.")
            path = self._ckpt_path(tag)

        blob = torch.load(path, map_location=self.device)
        # self.model.load_state_dict(blob["model_state"])
        # Load model weights safely
        if "model_state" in blob and blob["model_state"] is not None:
            _safe_model_state_load(self.model, blob["model_state"], strict=False)

        # Restore optimizer/scheduler only if present
        if blob.get("opt_state") is not None:
            if getattr(self.model, "opt", None) is None:
                # model builds optimizer lazily; ensure it exists
                if hasattr(self.model, "_build_or_reuse_optimizer"):
                    self.model._build_or_reuse_optimizer(initial=True)
            self.model.opt.load_state_dict(blob["opt_state"])
        if blob.get("sched_state") is not None and getattr(self.model, "lr_scheduler", None) is not None:
            try:
                self.model.lr_scheduler.load_state_dict(blob["sched_state"])
            except Exception:
                pass

        # Memory
        if blob.get("memory_state") is not None and hasattr(self.model.memory, "load_state_dict"):
            self.model.memory.load_state_dict(blob["memory_state"], device=self.device)

        # Trainer state
        ts = blob.get("trainer_state", {})
        self.global_task_counter = ts.get("global_task_counter", self.global_task_counter)
        self.test_acc_matrix = ts.get("test_acc_matrix", self.test_acc_matrix)
        if ts.get("log_path"):
            self.log_path = ts["log_path"]
        start_dsi = ts.get("dsi", 0)
        start_epoch = ts.get("epoch", 0)
        print(f"[CKPT] Loaded checkpoint from {path}; resume at ds={start_dsi}, epoch={start_epoch}")
        return int(start_dsi), int(start_epoch)

    # ---------------------- Training with save/resume ----------------------
    def train(self):
        self.model.train()
        log_interval = max(1, int(getattr(self.args, "log_interval", 10)))
        num_task_per_iter = max(1, int(getattr(self.args, "num_task_per_iter", 100)))
        num_eval_tasks = max(20, int(getattr(self.args, "num_eval_tasks", 100)))

        # Optional resume
        resume = bool(getattr(self.args, "resume", False))
        resume_tag = getattr(self.args, "resume_tag", None)
        resume_path = getattr(self.args, "resume_path", None)
        start_dsi, start_epoch = (0, 0)
        if resume:
            try:
                start_dsi, start_epoch = self.load_checkpoint(tag=resume_tag, path=resume_path)
            except Exception as e:
                print(f"[CKPT] Resume requested but failed to load: {e}. Starting fresh.")

        for dsi, ds_name in enumerate(self.order):
            if dsi < start_dsi:
                continue  # skip fully completed datasets
            ds_train = self.registry[ds_name]

            print(f"[ContinualMeta] Starting dataset '{ds_name}' as big task {dsi} "
                  f"for {self.epochs_per_dataset} epochs; "
                  f"{self.n_way}-way {self.k_shot}-shot, {self.q_query}-query; "
                  f"{num_task_per_iter} tasks/epoch.")

            

            for epoch in range(self.epochs_per_dataset):
                if dsi == start_dsi and epoch < start_epoch:
                    continue  # skip completed epochs within this dataset

                episodic_ds = EpisodicBatcher(
                    dataset=ds_train,
                    n_way=self.n_way,
                    k_shot=self.k_shot,
                    q_query=self.q_query,
                    num_tasks=num_task_per_iter,
                )
                trainloader = DataLoader(episodic_ds, batch_size=None)

                epoch_loss = 0.0
                for it, batch in enumerate(trainloader, start=1):
                    Sx, Sy, Qx, Qy, Sg, Qg = batch
                    Qx_b, Qy_b, dom_x, dom_y = self._episode_to_batch(Sx, Sy, Qx, Qy)
                    task_id_for_observe = dsi
                    loss = self.model.observe(
                        Qx_b, Qy_b, dom_x, dom_y, task_id_for_observe,
                        global_support_labels=Sg.to(self.device),
                        global_query_labels=Qg.to(self.device)
                    )
                    epoch_loss += float(loss)
                    if it % log_interval == 0:
                        print(f"[{ds_name}] epoch {epoch+1}/{self.epochs_per_dataset} "
                              f"iter {it}/{num_task_per_iter} loss={float(loss):.4f}")

                epoch_loss /= num_task_per_iter
                print(f"[{ds_name}] epoch {epoch+1} mean loss={epoch_loss:.4f}")

                # Validation after every epoch
                self._evaluate_validation_after_epoch(ds_name, num_eval_tasks=num_eval_tasks)

                # Save checkpoint per policy
                if self.ckpt_every_epochs > 0 and ((epoch + 1) % self.ckpt_every_epochs == 0):
                    self.save_checkpoint(dsi=dsi, epoch=epoch + 1, tag=f"ds{dsi}_ep{epoch+1}")

            # End-of-dataset boundary hook (only if we actually ran/finished this dataset now)
            # Boundary hook BEFORE dataset begins (only if starting new dataset or resuming exactly at its start)
            if self.args.known_boundary:
                if  hasattr(self.model.memory, "epoch_update"):
                    self.model.memory.epoch_update(task_counter=self.global_task_counter, model=self.model)
                    if hasattr(self.model, "task_boundary"):
                        self.model.task_boundary = True
                    if hasattr(self.model, "reset_state"):
                        with torch.no_grad():
                            x0 = ds_train.xs[:max(1, int(getattr(self.args, "batch_size", 32) // 2))]
                            x0 = x0.to(self.device)
                            self.model.reset_state(x0)
                    
            self.global_task_counter += 1

            # Evaluate lower triangle up to current dataset
            self._evaluate_test_lower_triangle(upto_idx=dsi, num_eval_tasks=num_eval_tasks)

            # Save a checkpoint at dataset boundary too
            self.save_checkpoint(dsi=dsi, epoch=self.epochs_per_dataset, tag=f"ds{dsi}_final")

        print("[ContinualMeta] Training done.")
        self._final_report_RA()