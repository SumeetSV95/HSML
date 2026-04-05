# prepare_mini_imagenet_from_gdrive_ac_cml.py
import argparse
from pathlib import Path
import tarfile
import shutil
import gdown
import torch
from PIL import Image
from torchvision import transforms as T

from ac_cml_transforms import enlist_transformation

def build_transform(cfg):
    return T.Compose(enlist_transformation(
        img_resize=cfg["img_resize"],
        is_grayscale=cfg["is_grayscale"],
        device=cfg["device"],
        img_normalise=cfg["mini_imagenet"]["img_normalise"],
        resize_interpolation=cfg["mini_imagenet"].get("resize_interpolation", "BILINEAR"),
    ))

def gdown_download(file_id: str, out_path: Path):
    url = f"https://drive.google.com/uc?id={file_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[miniImageNet] Downloading {file_id} -> {out_path}")
    gdown.download(url, str(out_path), quiet=False)

def extract_tar(tar_path: Path, dst_dir: Path):
    print(f"[miniImageNet] Extracting {tar_path} -> {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dst_dir)

def collect_images_from_split(root_dir: Path):
    class_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        cand = list(root_dir.rglob("*"))
        class_dirs = sorted([p for p in cand if p.is_dir() and any(x.suffix.lower() in (".jpg",".jpeg",".png") for x in p.glob("*.*"))])
    name_to_id = {d.name: i for i, d in enumerate(class_dirs)}
    items = []
    for d in class_dirs:
        cid = name_to_id[d.name]
        for img_path in d.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in (".jpg",".jpeg",".png"):
                items.append((img_path, cid))
    return items

def export_split_to_pt(split_dir: Path, out_pt: Path, tfm):
    items = collect_images_from_split(split_dir)
    if len(items) == 0:
        raise RuntimeError(f"No images found under {split_dir}")

    xs, ys = [], []
    for img_path, cid in items:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warn: failed {img_path}: {e}")
            continue
        x_t = tfm(img)
        xs.append(x_t.cpu())
        ys.append(cid)

    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y}, out_pt)
    print(f"[miniImageNet] Saved {out_pt} x={tuple(x.shape)} y={tuple(y.shape)}")

def maybe_collapse_one_level(dir_path: Path):
    children = [p for p in dir_path.iterdir() if p.is_dir()]
    if len(children) == 1 and any(q.is_dir() for q in children[0].iterdir()):
        nested = children[0]
        tmp = dir_path.parent / (dir_path.name + "_tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.move(str(nested), str(tmp))
        shutil.rmtree(dir_path)
        shutil.move(str(tmp), str(dir_path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--img_resize", type=int, default=28)
    ap.add_argument("--is_grayscale", action="store_true", default=False)
    ap.add_argument("--img_normalise", action="store_true", default=True)
    ap.add_argument("--train_id", type=str, default="107FTosYIeBn5QbynR46YG91nHcJ70whs")
    ap.add_argument("--val_id",   type=str, default="1hSMUMj5IRpf-nQs1OwgiQLmGZCN0KDWl")
    ap.add_argument("--test_id",  type=str, default="1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v")
    args = ap.parse_args()

    cfg = {
        "device": args.device,
        "img_resize": args.img_resize,
        "is_grayscale": args.is_grayscale,
        "mini_imagenet": {"img_normalise": args.img_normalise, "resize_interpolation": "BILINEAR"},
    }

    tfm = build_transform(cfg)

    work = Path(args.work_dir)
    out_root = Path(args.out_root)

    train_tar = work / "train.tar"
    val_tar   = work / "val.tar"
    test_tar  = work / "test.tar"
    gdown_download(args.train_id, train_tar)
    gdown_download(args.val_id, val_tar)
    gdown_download(args.test_id, test_tar)

    extract_root = work / "extracted"
    train_dir = extract_root / "train"
    val_dir   = extract_root / "val"
    test_dir  = extract_root / "test"
    extract_tar(train_tar, train_dir)
    extract_tar(val_tar, val_dir)
    extract_tar(test_tar, test_dir)

    for d in (train_dir, val_dir, test_dir):
        maybe_collapse_one_level(d)

    export_split_to_pt(train_dir, out_root / "mini_imagenet_train.pt", tfm)
    export_split_to_pt(val_dir,   out_root / "mini_imagenet_val.pt",   tfm)
    export_split_to_pt(test_dir,  out_root / "mini_imagenet_test.pt",  tfm)

    print("[miniImageNet] Done.")

if __name__ == "__main__":
    main()