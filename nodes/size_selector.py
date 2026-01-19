"""
Size Selector node for ComfyUI.
Provides common resolution presets for image and video generation.
"""

# Size presets: (width, height)
SIZE_PRESETS = {
    "Instagram Reel / TikTok (1080x1920)": (1080, 1920),
    "Instagram Square (1080x1080)": (1080, 1080),
    "Instagram Landscape (1080x566)": (1080, 566),
    "YouTube Shorts (1080x1920)": (1080, 1920),
    "720p Landscape (1280x720)": (1280, 720),
    "720p Portrait (720x1280)": (720, 1280),
    "1080p Landscape (1920x1080)": (1920, 1080),
    "1080p Portrait (1080x1920)": (1080, 1920),
    "4K Landscape (3840x2160)": (3840, 2160),
    "4K Portrait (2160x3840)": (2160, 3840),
    "Cinema 2.39:1 (1920x804)": (1920, 804),
    "Square 512 (512x512)": (512, 512),
    "Square 1024 (1024x1024)": (1024, 1024),
}


class SizeSelector:
    """
    Select from common resolution presets.

    Outputs width and height as separate integers for use with
    image/video generation nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(SIZE_PRESETS.keys()), {
                    "default": "1080p Landscape (1920x1080)",
                    "tooltip": "Select a resolution preset"
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "select"
    CATEGORY = "utils/latent-astronaut"

    def select(self, preset: str) -> tuple:
        width, height = SIZE_PRESETS[preset]
        return (width, height)


NODE_CLASS_MAPPINGS = {
    "SizeSelector": SizeSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SizeSelector": "Size Selector",
}
