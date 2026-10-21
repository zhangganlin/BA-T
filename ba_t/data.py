"""Load images for BA-T inference from image paths and/or folders.

Every image goes through the same deterministic preprocessing as the training /
evaluation dataloaders, with the principal point assumed at the image centre:

  1. crop to a window centred on the image centre (a no-op for even sizes);
  2. Lanczos-rescale so that the image just covers the target resolution
     (the short side becomes 224 for the released 224x224 model);
  3. centre-crop to the target resolution;
  4. normalize RGB to [-1, 1] (``ImgNorm``).

A view is a dict ``{'img': (3,H,W) float tensor, 'true_shape': (2,) int32 tensor
(H, W), 'path': str, 'crop': dict}``; ``crop`` records the transform so that
pixels can be mapped back to the original image (``to_original_pixels``).
``collate_views`` stacks views into the batched dicts that ``BAT.forward``
expects.
"""
import os
import re

import numpy as np
import PIL.Image
import PIL.ImageOps
import torch

from .utils.cropping import rescale_image, crop_image, center_crop_bbox
from .utils.image import ImgNorm

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
DEFAULT_RESOLUTION = (224, 224)  # (W, H) the released model was trained at


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def collect_image_paths(inputs):
    """Expand a path or a list of paths (image files and/or folders) into a list of image files.

    Folders are scanned non-recursively and sorted in natural order (frame2 < frame10);
    explicitly listed files keep the given order.
    """
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    paths = []
    for p in inputs:
        p = os.fspath(p)
        if os.path.isdir(p):
            files = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(IMG_EXTENSIONS)]
            if not files:
                raise FileNotFoundError(f'no images ({", ".join(IMG_EXTENSIONS)}) found in folder {p}')
            paths += sorted(files, key=_natural_key)
        elif os.path.isfile(p):
            paths.append(p)
        else:
            raise FileNotFoundError(f'{p} is neither an image file nor a folder')
    return paths


def _parse_resolution(resolution):
    if isinstance(resolution, int):
        return resolution, resolution
    width, height = resolution
    assert width % 16 == 0 and height % 16 == 0, f'resolution {resolution} must be a multiple of the patch size 16'
    return int(width), int(height)


def preprocess_image(image, resolution=DEFAULT_RESOLUTION):
    """Crop + resize a PIL image like the training dataloader (deterministic, no augmentation).

    Returns the processed PIL image and a dict describing the transform.
    """
    width, height = _parse_resolution(resolution)
    image = image.convert('RGB')
    W, H = image.size

    # 1. crop centred on the principal point (assumed at the image centre)
    cx, cy = int(W/2), int(H/2)
    min_margin_x = min(cx, W-cx)
    min_margin_y = min(cy, H-cy)
    l1, t1 = cx - min_margin_x, cy - min_margin_y
    image = crop_image(image, (l1, t1, cx + min_margin_x, cy + min_margin_y))

    # the target resolution is landscape; use it transposed for portrait images
    target = (width, height)
    if width != height and image.size[1] > 1.1 * image.size[0]:
        target = (height, width)

    # 2. high-quality Lanczos down-scaling, 3. centre crop
    image, scale = rescale_image(image, target)
    l2, t2, r2, b2 = center_crop_bbox(image.size, target)
    image = crop_image(image, (l2, t2, r2, b2))

    crop = {'orig_size': (W, H), 'offset1': (int(l1), int(t1)), 'scale': float(scale),
            'offset2': (int(l2), int(t2)), 'size': image.size}
    return image, crop


def to_original_pixels(uv, crop):
    """Map pixel coordinates (..., 2) [u=column, v=row] of a preprocessed image back to
    the original image (pixel centres at integer coordinates)."""
    uv = np.asarray(uv, dtype=np.float64)
    off2 = np.asarray(crop['offset2'], dtype=np.float64)
    off1 = np.asarray(crop['offset1'], dtype=np.float64)
    return (uv + off2 + 0.5) / crop['scale'] - 0.5 + off1


def load_image(path, resolution=DEFAULT_RESOLUTION):
    """Read one image file (EXIF orientation applied) and preprocess it into a view dict."""
    with PIL.Image.open(path) as im:
        image = PIL.ImageOps.exif_transpose(im).convert('RGB')
    image, crop = preprocess_image(image, resolution)
    width, height = image.size
    return {'img': ImgNorm(image),
            'true_shape': torch.tensor((height, width), dtype=torch.int32),
            'path': os.fspath(path),
            'crop': crop}


class ImageViews(torch.utils.data.Dataset):
    """Dataset over image paths / folders, yielding preprocessed view dicts."""

    def __init__(self, inputs, resolution=DEFAULT_RESOLUTION):
        self.paths = collect_image_paths(inputs)
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return load_image(self.paths[idx], self.resolution)


def collate_views(views):
    """Stack a list of view dicts into one batched view: 'img' (B,3,H,W), 'true_shape' (B,2).

    All views must have the same shape (always true for square resolutions).
    """
    return {'img': torch.stack([v['img'] for v in views]),
            'true_shape': torch.stack([v['true_shape'] for v in views]),
            'path': [v['path'] for v in views],
            'crop': [v['crop'] for v in views]}


def load_views(inputs, resolution=DEFAULT_RESOLUTION):
    """Load images from paths / folders as a list of single-image batched views (B=1)."""
    ds = ImageViews(inputs, resolution)
    return [collate_views([ds[i]]) for i in range(len(ds))]


def denormalize(img):
    """(3,H,W) or (B,3,H,W) tensor in [-1,1] -> uint8 numpy (H,W,3) or (B,H,W,3)."""
    img = ((img.detach().cpu().float() + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return img.movedim(-3, -1).numpy()
