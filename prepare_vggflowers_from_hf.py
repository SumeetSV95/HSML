# prepare_vggflowers_from_hf_ac_cml.py
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from PIL import Image, ImageOps

from ac_cml_transforms import enlist_transformation
from torchvision import transforms as T
from io import BytesIO
import os

def build_transform(cfg, task_key="vggflowers"):
    tfms = enlist_transformation(
        img_resize=cfg["img_resize"],
        is_grayscale=cfg["is_grayscale"],
        device=cfg["device"],
        img_normalise=cfg[task_key]["img_normalise"],
        resize_interpolation=cfg[task_key].get("resize_interpolation", "BILINEAR"),
    )
    return T.Compose(tfms)

def safe_exif_transpose(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def to_pil(image_field):
    # Already a PIL image
    if isinstance(image_field, Image.Image):
        return safe_exif_transpose(image_field)

    # dict with possible keys: "bytes", "path"
    if isinstance(image_field, dict):
        if "bytes" in image_field and image_field["bytes"] is not None:
            img = Image.open(BytesIO(image_field["bytes"]))
            return safe_exif_transpose(img)
        if "path" in image_field and image_field["path"]:
            img = Image.open(image_field["path"])
            return safe_exif_transpose(img)

    # raw bytes
    if isinstance(image_field, (bytes, bytearray)):
        img = Image.open(BytesIO(image_field))
        return safe_exif_transpose(img)

    # file path string
    if isinstance(image_field, str) and os.path.exists(image_field):
        img = Image.open(image_field)
        return safe_exif_transpose(img)

    # Last resort: try to let PIL open whatever it is (may throw)
    img = Image.open(image_field)
    return safe_exif_transpose(img)

def export_split_hf(out_dir: Path, split: str, tfm):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load without casting/overriding features; iterate and decode manually
    ds = load_dataset("GATE-engine/vggflowers", split=split)

    xs, ys = [], []
    for ex in ds:
        pil_img = to_pil(ex["image"])
        # ensure mode matches your pipeline expectations
        if pil_img.mode not in ("RGB", "L"):
            pil_img = pil_img.convert("RGB")
        y = int(ex["label"])

        x_t = tfm(pil_img)
        xs.append(x_t.cpu())
        ys.append(y)

    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    out_pt = out_dir / f"vggflowers_{'val' if split=='validation' else split}.pt"
    torch.save({"x": x, "y": y}, out_pt)
    print(f"[VGG-Flowers] Saved {out_pt} x={tuple(x.shape)} y={tuple(y.shape)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--img_resize", type=int, default=28)
    ap.add_argument("--is_grayscale", action="store_true", default=False)
    ap.add_argument("--img_normalise", action="store_true", default=True)
    args = ap.parse_args()

    cfg = {
        "device": args.device,
        "img_resize": args.img_resize,
        "is_grayscale": args.is_grayscale,
        "vggflowers": {"img_normalise": args.img_normalise, "resize_interpolation": "BILINEAR"},
    }

    tfm = build_transform(cfg, "vggflowers")
    out_dir = Path(args.out_root)

    for split in ("train", "validation", "test"):
        export_split_hf(out_dir, split, tfm)

if __name__ == "__main__":
    main()