"""
Image batch utility nodes for ComfyUI.
"""

import torch


class BatchLastImage:
    """
    Get the last image from a batch.
    Useful for extracting the final frame from a loaded video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "get_last"
    CATEGORY = "image/latent-astronaut"

    def get_last(self, images: torch.Tensor) -> tuple:
        # Keep batch dimension with [-1:]
        return (images[-1:],)


class VideoLengthFromBatch:
    """
    Calculate video duration from a batch of images and frame rate.
    Returns the length in seconds.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {
                    "default": 24.0,
                    "min": 0.1,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second"
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("seconds", "frame_count")
    FUNCTION = "calculate"
    CATEGORY = "image/latent-astronaut"

    def calculate(self, images: torch.Tensor, frame_rate: float) -> tuple:
        frame_count = images.shape[0]
        seconds = frame_count / frame_rate
        return (seconds, frame_count)


NODE_CLASS_MAPPINGS = {
    "BatchLastImage": BatchLastImage,
    "VideoLengthFromBatch": VideoLengthFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchLastImage": "Batch Last Image",
    "VideoLengthFromBatch": "Video Length from Batch",
}
