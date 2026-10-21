# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# cropping utilities (image-only versions of the training-time crop / resize)
# --------------------------------------------------------
import numpy as np
import PIL.Image

try:
    lanczos = PIL.Image.Resampling.LANCZOS
except AttributeError:
    lanczos = PIL.Image.LANCZOS


def rescale_image(image, output_resolution):
    """ Rescale an image with Lanczos interpolation so that (out_width, out_height) >= output_resolution.
        Returns the rescaled image and the scale factor.
    """
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / input_resolution) + 1e-8
    output_resolution = np.floor(input_resolution * scale_final).astype(int)
    image = image.resize(tuple(int(x) for x in output_resolution), resample=lanczos)
    return image, scale_final


def crop_image(image, crop_bbox):
    """ Return a crop (l, t, r, b) of the input image.
    """
    l, t, r, b = (int(x) for x in crop_bbox)
    return image.crop((l, t, r, b))


def center_crop_bbox(input_size, output_resolution):
    """ Crop box of size output_resolution centred in an image of size input_size (both (W,H)).
    """
    curr_W, curr_H = input_size
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(curr_W/2 - out_width/2)), np.int32(np.round(curr_H/2 - out_height/2))
    return (l, t, l+out_width, t+out_height)
