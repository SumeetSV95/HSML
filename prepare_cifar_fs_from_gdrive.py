# prepare_cifar_fs_from_gdrive_ac_cml.py
import argparse
from pathlib import Path
import zipfile
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
        img_normalise=cfg["cifar_fs"]["img_normalise"],
        resize_interpolation=cfg["cifar_fs"].get("resize_interpolation", "BILINEAR"),
    ))

def gdown_download(file_id: str, out_path: Path):
    url = f"https://drive.google.com/uc?id={file_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[CIFAR-FS] Downloading {file_id} -> {out_path}")
    gdown.download(url, str(out_path), quiet=False)

def extract_zip(zip_path: Path, dst_dir: Path):
    print(f"[CIFAR-FS] Extracting {zip_path} -> {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dst_dir)

def read_class_list(split_txt: Path):
    with open(split_txt, "r") as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    return classes

def collect_images_for_classes(data_root: Path, class_names):
    items = []
    for ci, cname in enumerate(class_names):
        class_dir = data_root / cname
        if not class_dir.exists():
            # Try fuzzy match
            alts = [p for p in data_root.glob(f"*{cname}*") if p.is_dir()]
            class_dir = alts[0] if alts else None
        if class_dir is None or not class_dir.exists():
            print(f"Warn: missing class folder: {cname}")
            continue
        for img_path in class_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                items.append((img_path, ci))
    return items

def export_split_to_pt(data_root: Path, split_file: Path, out_pt: Path, tfm):
    class_names = read_class_list(split_file)
    items = collect_images_for_classes(data_root, class_names)
    if len(items) == 0:
        raise RuntimeError(f"No images found for {split_file}")

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
    print(f"[CIFAR-FS] Saved {out_pt} x={tuple(x.shape)} y={tuple(y.shape)} (classes={len(set(ys))})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--img_resize", type=int, default=28)
    ap.add_argument("--is_grayscale", action="store_true", default=False)
    ap.add_argument("--img_normalise", action="store_true", default=True)
    ap.add_argument("--zip_id", type=str, default="1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI")
    args = ap.parse_args()

    cfg = {
        "device": args.device,
        "img_resize": args.img_resize,
        "is_grayscale": args.is_grayscale,
        "cifar_fs": {"img_normalise": args.img_normalise, "resize_interpolation": "BILINEAR"},
    }

    tfm = build_transform(cfg)

    work = Path(args.work_dir)
    out_root = Path(args.out_root)

    zip_path = work / "cifar_fs.zip"
    gdown_download(args.zip_id, zip_path)

    extract_root = work / "extracted"
    extract_zip(zip_path, extract_root)

    cifar100_root = extract_root / "cifar100"
    if not cifar100_root.exists():
        candidates = list(extract_root.rglob("cifar100"))
        if not candidates:
            raise RuntimeError("cifar100 folder not found after extraction.")
        cifar100_root = candidates[0]

    data_root = cifar100_root / "data"
    splits_root = cifar100_root / "splits" / "bertinetto"

    out_root.mkdir(parents=True, exist_ok=True)

    export_split_to_pt(data_root, splits_root / "train.txt", out_root / "cifar_fs_train.pt", tfm)
    export_split_to_pt(data_root, splits_root / "val.txt",   out_root / "cifar_fs_val.pt",   tfm)
    export_split_to_pt(data_root, splits_root / "test.txt",  out_root / "cifar_fs_test.pt",  tfm)

    print("[CIFAR-FS] Done.")

if __name__ == "__main__":
    main()