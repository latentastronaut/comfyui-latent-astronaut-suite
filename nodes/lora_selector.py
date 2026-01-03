"""
LoRA Selector node for ComfyUI.
Select one LoRA from a list based on an index input.
"""

import folder_paths
import comfy.utils
import comfy.sd

MAX_LORA_SLOTS = 16


class LoraLoaderModelOnlySelector:
    """
    Select and load one LoRA from up to 16 choices based on an index.
    Useful for batch processing or iterating through LoRAs.
    LoRA slots appear dynamically as you set them.
    Use the Reset All button to clear all LoRA selections.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        lora_list_with_none = ["None"] + lora_list

        inputs = {
            "required": {
                "model": ("MODEL",),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_LORA_SLOTS - 1,
                    "tooltip": "Which LoRA to load (0-indexed)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "LoRA strength"
                }),
            },
            "optional": {},
        }

        for i in range(MAX_LORA_SLOTS):
            inputs["optional"][f"lora_{i}"] = (lora_list_with_none, {"default": "None"})

        return inputs

    RETURN_TYPES = ("MODEL", "STRING", "INT")
    RETURN_NAMES = ("model", "lora_name", "count")
    OUTPUT_TOOLTIPS = (
        "Model with LoRA applied",
        "Name of loaded LoRA (or 'None')",
        "Number of non-None LoRAs configured",
    )
    FUNCTION = "load_lora"
    CATEGORY = "loaders/latent-astronaut"

    def load_lora(self, model, index, strength, **kwargs):
        # Collect all LoRAs
        loras = [kwargs.get(f"lora_{i}", "None") for i in range(MAX_LORA_SLOTS)]

        # Count non-None entries
        count = sum(1 for l in loras if l != "None")

        # Clamp index
        index = max(0, min(index, MAX_LORA_SLOTS - 1))
        lora_name = loras[index]

        # If None or zero strength, return unchanged
        if lora_name == "None" or strength == 0:
            return (model, lora_name, count)

        # Load the LoRA
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return (model_lora, lora_name, count)


class LoraLoaderSelector:
    """
    Select and load one LoRA from up to 16 choices based on an index.
    This version also modifies CLIP.
    LoRA slots appear dynamically as you set them.
    Use the Reset All button to clear all LoRA selections.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        lora_list_with_none = ["None"] + lora_list

        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_LORA_SLOTS - 1,
                    "tooltip": "Which LoRA to load (0-indexed)"
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                }),
            },
            "optional": {},
        }

        for i in range(MAX_LORA_SLOTS):
            inputs["optional"][f"lora_{i}"] = (lora_list_with_none, {"default": "None"})

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "INT")
    RETURN_NAMES = ("model", "clip", "lora_name", "count")
    OUTPUT_TOOLTIPS = (
        "Model with LoRA applied",
        "CLIP with LoRA applied",
        "Name of loaded LoRA (or 'None')",
        "Number of non-None LoRAs configured",
    )
    FUNCTION = "load_lora"
    CATEGORY = "loaders/latent-astronaut"

    def load_lora(self, model, clip, index, strength_model, strength_clip, **kwargs):
        loras = [kwargs.get(f"lora_{i}", "None") for i in range(MAX_LORA_SLOTS)]

        # Count non-None entries
        count = sum(1 for l in loras if l != "None")

        index = max(0, min(index, MAX_LORA_SLOTS - 1))
        lora_name = loras[index]

        if lora_name == "None" or (strength_model == 0 and strength_clip == 0):
            return (model, clip, lora_name, count)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora, lora_name, count)


NODE_CLASS_MAPPINGS = {
    "LoraLoaderModelOnlySelector": LoraLoaderModelOnlySelector,
    "LoraLoaderSelector": LoraLoaderSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderModelOnlySelector": "LoRA Loader Model Only Selector",
    "LoraLoaderSelector": "LoRA Loader Selector",
}
