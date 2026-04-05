import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class RotatedMNISTDataset(Dataset):
    """Sequential Rotated-MNIST episodes compatible with VC-BML pipeline.

    The dataset exposes 5-way tasks ordered by rotation angle. For each
    rotation we carve out ``num_shot`` support and ``num_query`` query
    examples per digit (default digits 0-4). Samples are arranged so that the
    sequential sampler (``seqtask=True``) yields tasks in rotation order,
    while evaluation can still draw random episodic batches via the existing
    sampler utilities.
    """

    def __init__(
        self,
        pt_file: str,
        split: str,
        transform=None,
        device: Optional[str] = None,
        cuda_img_tensor: bool = True,
        digits: Optional[Sequence[int]] = None,
        num_way: int = 5,
        num_shot: int = 5,
        num_query: int = 15,
        verbose: Optional[str] = None,
    ) -> None:
        super().__init__()

        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.transform = transform
        self.device = device
        self.cuda_img_tensor = cuda_img_tensor
        self.verbose = verbose
        self.seqtask = True  # flag consumed in training loop

        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query

        if digits is None:
            digits = list(range(num_way))
        else:
            digits = list(digits)
        if len(digits) < num_way:
            raise ValueError(
                f"Need at least {num_way} digits but received {len(digits)}"
            )
        self.digits = digits[:num_way]

        pt_path = self._resolve_path(pt_file)
        train_tasks, test_tasks = torch.load(pt_path, map_location="cpu")
        tasks = train_tasks if split == "train" else test_tasks

        # Sort tasks by rotation (angle) to match FTML "normal" ordering.
        tasks_sorted = sorted(tasks, key=lambda item: float(item[0]))
        self.angles: List[float] = [float(item[0]) for item in tasks_sorted]
        self.seqtask_num_batch = len(self.angles)

        # Build dataframe records with deterministic support/query chunks.
        records: List[dict] = []
        support_records_by_class: List[List[dict]] = [[] for _ in range(self.num_way)]
        query_records_by_class: List[List[dict]] = [[] for _ in range(self.num_way)]

        for rotation_index, (angle, images, labels) in enumerate(tasks_sorted):
            imgs_np = images.cpu().numpy() if isinstance(images, torch.Tensor) else images
            lbls_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

            for local_cls, digit in enumerate(self.digits):
                digit_mask = lbls_np == digit
                digit_imgs = imgs_np[digit_mask]
                if digit_imgs.shape[0] < (self.num_shot + self.num_query):
                    raise ValueError(
                        f"Rotation {rotation_index} digit {digit} has only {digit_imgs.shape[0]} samples"
                    )

                support_imgs = digit_imgs[: self.num_shot]
                query_imgs = digit_imgs[self.num_shot : self.num_shot + self.num_query]

                support_records_by_class[local_cls].extend(
                    self._build_records(
                        tensors=support_imgs,
                        cls_lbl=local_cls,
                        cls_name=f"digit_{digit}",
                        supercls=f"rot_{rotation_index:02d}",
                        angle=float(angle),
                        rotation_index=rotation_index,
                        split="support",
                    )
                )

                query_records_by_class[local_cls].extend(
                    self._build_records(
                        tensors=query_imgs,
                        cls_lbl=local_cls,
                        cls_name=f"digit_{digit}",
                        supercls=f"rot_{rotation_index:02d}",
                        angle=float(angle),
                        rotation_index=rotation_index,
                        split="query",
                    )
                )

        for local_cls in range(self.num_way):
            # ensure support precedes query so sampler array_split matches rotation order
            records.extend(support_records_by_class[local_cls])
            records.extend(query_records_by_class[local_cls])

        self.df = pd.DataFrame(records)
        self._relabel = None

        # Pre-compute per-class indices for sequential sampling.
        self.subset_indices_per_cls = [
            (cls_lbl, self.df[self.df["cls_lbl"] == cls_lbl].index.to_list())
            for cls_lbl in range(self.num_way)
        ]

    @staticmethod
    def _resolve_path(pt_file: str) -> str:
        if os.path.isabs(pt_file):
            return pt_file
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate = os.path.join(repo_root, pt_file)
        return os.path.normpath(candidate)

    def _build_records(
        self,
        tensors: np.ndarray,
        cls_lbl: int,
        cls_name: str,
        supercls: str,
        angle: float,
        rotation_index: int,
        split: str,
    ) -> List[dict]:
        records = []
        for array in tensors:
            image_np = array.reshape(28, 28)
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8), mode="L")
            if self.transform is not None:
                processed = self.transform(pil_image)
            else:
                processed = torch.from_numpy(image_np).unsqueeze(0).float()
            if self.cuda_img_tensor and isinstance(processed, torch.Tensor):
                processed = processed.to(device=self.device)
            records.append(
                {
                    "supercls": supercls,
                    "cls_name": f"{supercls}.{cls_name}",
                    "cls_lbl": cls_lbl,
                    "angle": angle,
                    "rotation_index": rotation_index,
                    "split": split,
                    "cuda_tensor": processed,
                }
            )
        return records

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.df.loc[idx, "cuda_tensor"]
        label = torch.tensor(self.df.loc[idx, "cls_lbl"], device=self.device)
        return image, label

    @property
    def relabel(self) -> Optional[Tuple[str, Sequence[str]]]:
        return self._relabel

    @relabel.setter
    def relabel(self, labels: Optional[Tuple[str, Sequence[str]]]) -> None:
        self._relabel = labels

    def relbl_df(self) -> None:
        if self._relabel is None:
            return
        column, values = self._relabel
        for ind, value in enumerate(values):
            self.df.loc[self.df[column] == value, "cls_lbl"] = ind

