# prepare_omniglot_from_gdrive_ac_cml.py
import argparse
from pathlib import Path
import zipfile
import gdown
import torch
from PIL import Image
from torchvision import transforms as T

# AC-CML helpers
class ToDevice(object):
    def __init__(self, device=None):
        self.device = device
    def __call__(self, tensor_pic):
        return tensor_pic.to(self.device)

class NormaliseMinMax(object):
    def __call__(self, tensor):
        minv = torch.min(tensor)
        maxv = torch.max(tensor)
        denom = (maxv - minv).clamp_min(1e-6)
        return (tensor - minv) / denom

class RepeatChannels(object):
    def __init__(self, out_channels=3):
        self.out_channels = out_channels
    def __call__(self, tensor):
        # tensor: (C,H,W). If C==1, repeat to out_channels; if already 3, return as-is.
        if tensor.dim() == 3 and tensor.size(0) == 1 and self.out_channels > 1:
            return tensor.repeat(self.out_channels, 1, 1)
        return tensor

def enlist_transformation(img_resize: int, device: str, img_normalise: bool,
                          resize_interpolation: str = "BILINEAR",
                          force_grayscale: bool = True,
                          repeat_to_3ch: bool = True):
    from PIL import Image as PILImage
    interp = getattr(PILImage, resize_interpolation.upper(), PILImage.BILINEAR)
    tfms = []
    if force_grayscale:
        tfms.append(T.Grayscale(num_output_channels=1))
    tfms.append(T.Resize((img_resize, img_resize), interpolation=interp))
    tfms.append(T.ToTensor())
    if img_normalise:
        tfms.append(NormaliseMinMax())
    if repeat_to_3ch:
        tfms.append(RepeatChannels(3))
    tfms.append(ToDevice(device=device))
    return tfms

# Superclass splits
TRAIN_ALPHABETS = [
    "Alphabet_of_the_Magi","Angelic","Armenian","Asomtavruli_(Georgian)","Atlantean","Aurek-Besh","Avesta","Balinese","Bengali",
    "Braille","Burmese_(Myanmar)","Early_Aramaic","Grantha","Gujarati","Gurmukhi","Hebrew","Inuktitut_(Canadian_Aboriginal_Syllabics)",
    "Japanese_(hiragana)","Japanese_(katakana)","Kannada","Keble","Korean","Latin","Malayalam","Malay_(Jawi_-_Arabic)",
    "Manipuri","Mongolian","Ojibwe_(Canadian_Aboriginal_Syllabics)","Old_Church_Slavonic_(Cyrillic)","Oriya","Sanskrit","Sylheti",
    "Tengwar","Tifinagh","ULOG",
]
VAL_ALPHABETS = [
    "Anglo-Saxon_Futhorc","Arcadian","Blackfoot_(Canadian_Aboriginal_Syllabics)","Cyrillic","Ge_ez","Glagolitic","N_Ko",
]
TEST_ALPHABETS = [
    "Atemayar_Qelisayer","Futurama","Greek","Mkhedruli_(Georgian)","Syriac_(Estrangelo)","Syriac_(Serto)","Tagalog","Tibetan",
]

def build_transform(device: str, img_resize: int, img_normalise: bool):
    return T.Compose(enlist_transformation(
        img_resize=img_resize,
        device=device,
        img_normalise=img_normalise,
        resize_interpolation="BILINEAR",
        force_grayscale=True,     # keep Omniglot semantics
        repeat_to_3ch=True        # NEW: repeat to 3 channels for 3-ch model
    ))

def gdown_download(file_id: str, out_path: Path):
    url = f"https://drive.google.com/uc?id={file_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Omniglot] Downloading {file_id} -> {out_path}")
    gdown.download(url, str(out_path), quiet=False)

def extract_zip(zip_path: Path, dst_dir: Path):
    print(f"[Omniglot] Extracting {zip_path} -> {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dst_dir)

def list_alphabets(data_root: Path):
    return sorted([p.name for p in data_root.iterdir() if p.is_dir()])

def collect_items_for_alphabets(data_root: Path, alphabets):
    items = []
    for alpha in alphabets:
        alpha_dir = data_root / alpha
        if not alpha_dir.exists():
            print(f"Warn: missing alphabet {alpha}")
            continue
        for char_dir in alpha_dir.iterdir():
            if not char_dir.is_dir():
                continue
            class_name = f"{alpha}/{char_dir.name}"
            for img_path in char_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in (".png",".jpg",".jpeg",".bmp"):
                    items.append((img_path, class_name))
    return items

def build_class_id_map(items):
    class_names = sorted({cn for _, cn in items})
    return {cn: i for i, cn in enumerate(class_names)}

def export_split_to_pt(data_root: Path, alphabets, out_pt: Path, tfm):
    items = collect_items_for_alphabets(data_root, alphabets)
    if len(items) == 0:
        raise RuntimeError(f"No images found for provided alphabets under {data_root}")

    cls_to_id = build_class_id_map(items)
    xs, ys = [], []
    for img_path, class_name in items:
        try:
            # open as grayscale; transform will convert to 1ch tensor then repeat to 3ch
            img = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"Warn: failed {img_path}: {e}")
            continue
        x_t = tfm(img)
        xs.append(x_t.cpu())
        ys.append(cls_to_id[class_name])

    x = torch.stack(xs, dim=0)              # (N, 3, H, W) after RepeatChannels
    y = torch.tensor(ys, dtype=torch.long)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y}, out_pt)
    print(f"[Omniglot] Saved {out_pt} x={tuple(x.shape)} y={tuple(y.shape)} (classes={len(cls_to_id)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--img_resize", type=int, default=28)
    ap.add_argument("--img_normalise", action="store_true", default=True)
    ap.add_argument("--zip_id", type=str, default="10ml4OJRc13pl5Ms3mm2VyscyTj94c87O")
    args = ap.parse_args()

    tfm = build_transform(device=args.device, img_resize=args.img_resize, img_normalise=args.img_normalise)

    work = Path(args.work_dir)
    out_root = Path(args.out_root)

    zip_path = work / "omniglot.zip"
    gdown_download(args.zip_id, zip_path)

    extract_root = work / "extracted"
    extract_zip(zip_path, extract_root)

    omni_root = extract_root / "omniglot"
    data_root = omni_root / "data"
    if not data_root.exists():
        candidates = list(extract_root.rglob("omniglot"))
        if not candidates:
            raise RuntimeError("omniglot folder not found.")
        omni_root = candidates[0]
        data_root = omni_root / "data"

    # Optional sanity
    have = set(list_alphabets(data_root))
    need = set(TRAIN_ALPHABETS) | set(VAL_ALPHABETS) | set(TEST_ALPHABETS)
    miss = sorted(list(need - have))
    if miss:
        print(f"Warning: missing alphabets {miss}")

    out_root.mkdir(parents=True, exist_ok=True)
    export_split_to_pt(data_root, TRAIN_ALPHABETS, out_root / "omniglot_train.pt", tfm)
    export_split_to_pt(data_root, VAL_ALPHABETS,   out_root / "omniglot_val.pt",   tfm)
    export_split_to_pt(data_root, TEST_ALPHABETS,  out_root / "omniglot_test.pt",  tfm)

    print("[Omniglot] Done. Tensors are 3-channel to match num_in_ch=3.")

if __name__ == "__main__":
    main()