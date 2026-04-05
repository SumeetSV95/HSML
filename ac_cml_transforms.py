# ac_cml_transforms.py

import torch
from typing import List
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class ToDevice(object):
    def __init__(self, device=None):
        self.device = device

    def __call__(self, tensor_pic):
        return tensor_pic.to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + f'(device={self.device})'

class NormaliseMinMax(object):
    def __call__(self, tensor):
        minv = torch.min(tensor)
        maxv = torch.max(tensor)
        # Guard against degenerate images
        denom = (maxv - minv).clamp_min(1e-6)
        return (tensor - minv) / denom

    def __repr__(self):
        return self.__class__.__name__ + '(per-image min-max 0..1)'

class ChangeBlackToColour(object):
    def __init__(self, colour):
        # colour: list/tuple of 3 floats in [0,1]
        self.colour = torch.tensor(colour, dtype=torch.float32)

    def __call__(self, tensor):
        # tensor: (C,H,W), typically C=1 or 3
        # Define “black” as 1. per your original code; adapt if needed
        mask = (tensor == 1.0)
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        # Broadcast colour to HxW
        coloured = tensor.clone()
        for c in range(3):
            coloured[c][mask[0]] = self.colour[c]
        return coloured

    def __repr__(self):
        return self.__class__.__name__ + f'(rgb_colour={self.colour.tolist()})'

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image):  # PIL Image
        return TF.rotate(image, self.angle)

    def __repr__(self):
        return self.__class__.__name__ + f'(angle={self.angle})'

def enlist_transformation(img_resize: int, is_grayscale: bool, device: str, img_normalise: bool,
                          resize_interpolation: str = "BILINEAR",
                          extra_aug: List = None):
    """
    Builds a list of transforms matching AC-CML snippet.
    Order:
      - [Optional] To Grayscale
      - Resize
      - ToTensor
      - [Optional] NormaliseMinMax (per image)
      - [Optional] extra_aug (e.g., Rotate)
      - ToDevice
    """
    tfms = []
    if is_grayscale:
        tfms.append(T.Grayscale(num_output_channels=1))

    interp = getattr(Image, resize_interpolation.upper(), Image.BILINEAR)
    tfms.append(T.Resize((img_resize, img_resize), interpolation=interp))
    tfms.append(T.ToTensor())

    if img_normalise:
        tfms.append(NormaliseMinMax())

    if extra_aug:
        tfms.extend(extra_aug)

    tfms.append(ToDevice(device=device))
    return tfms