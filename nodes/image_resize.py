"""
Image Resize to Total Pixels node for ComfyUI.
Resizes images to a target total pixel count with dimension constraints.
"""

import torch
import numpy as np
from PIL import Image


INTERPOLATION_METHODS = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}


class ImageResizeToTotalPixels:
    """
    Resize an image to a target total pixel count.
    Dimensions are constrained to multiples of the specified increment.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Target size in megapixels (1.0 = 1,000,000 pixels)"
                }),
                "increment": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                    "tooltip": "Dimensions will be multiples of this value"
                }),
                "resize_method": (["stretch", "crop"], {
                    "default": "stretch",
                    "tooltip": "Stretch distorts aspect ratio; crop maintains it"
                }),
                "interpolation": (list(INTERPOLATION_METHODS.keys()), {
                    "default": "lanczos",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "image/latent-astronaut"

    def resize(self, image: torch.Tensor, megapixels: float, increment: int,
               resize_method: str, interpolation: str) -> tuple:
        batch_size, orig_h, orig_w, channels = image.shape
        target_pixels = int(megapixels * 1_000_000)

        orig_aspect = orig_w / orig_h
        interp_mode = INTERPOLATION_METHODS[interpolation]

        # Calculate dimensions that hit target pixels while maintaining aspect ratio
        # pixels = w * h, aspect = w / h, so w = sqrt(pixels * aspect), h = sqrt(pixels / aspect)
        target_h = int(np.sqrt(target_pixels / orig_aspect))
        target_w = int(np.sqrt(target_pixels * orig_aspect))

        # Round to increment
        target_w = max(increment, round(target_w / increment) * increment)
        target_h = max(increment, round(target_h / increment) * increment)

        if resize_method == "stretch":
            # Direct resize to target dimensions (may distort aspect ratio)
            final_w, final_h = target_w, target_h
            result = self._batch_resize(image, final_w, final_h, interp_mode)

        else:  # crop
            # Resize so image covers target dimensions, then center crop
            # Scale factor needed to cover the target area
            scale_w = target_w / orig_w
            scale_h = target_h / orig_h
            scale = max(scale_w, scale_h)

            # Intermediate size (will be >= target in both dimensions)
            inter_w = int(np.ceil(orig_w * scale))
            inter_h = int(np.ceil(orig_h * scale))

            # Resize to intermediate size
            resized = self._batch_resize(image, inter_w, inter_h, interp_mode)

            # Center crop to target
            result = self._center_crop(resized, target_w, target_h)
            final_w, final_h = target_w, target_h

        return (result, final_w, final_h)

    def _batch_resize(self, images: torch.Tensor, width: int, height: int,
                      interp_mode: int) -> torch.Tensor:
        """Resize a batch of images using PIL."""
        results = []
        for i in range(images.shape[0]):
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img = pil_img.resize((width, height), interp_mode)
            result_np = np.array(pil_img).astype(np.float32) / 255.0
            results.append(result_np)
        return torch.from_numpy(np.stack(results))

    def _center_crop(self, images: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """Center crop a batch of images."""
        _, img_h, img_w, _ = images.shape

        # Calculate crop offsets
        left = (img_w - width) // 2
        top = (img_h - height) // 2

        return images[:, top:top + height, left:left + width, :]


NODE_CLASS_MAPPINGS = {
    "ImageResizeToTotalPixels": ImageResizeToTotalPixels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResizeToTotalPixels": "Image Resize to Total Pixels",
}
