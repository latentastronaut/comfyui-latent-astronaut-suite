"""
ComfyUI Latent Astronaut Suite - Custom nodes for ComfyUI.

Provides loop nodes, string list utilities, and LoRA selector.
"""

# Loop nodes
from .nodes.for_loop import NODE_CLASS_MAPPINGS as FOR_LOOP_MAPPINGS
from .nodes.for_loop import NODE_DISPLAY_NAME_MAPPINGS as FOR_LOOP_NAMES

# String list utilities
from .nodes.string_list import NODE_CLASS_MAPPINGS as STRING_LIST_MAPPINGS
from .nodes.string_list import NODE_DISPLAY_NAME_MAPPINGS as STRING_LIST_NAMES

# LoRA selector
from .nodes.lora_selector import NODE_CLASS_MAPPINGS as LORA_SELECTOR_MAPPINGS
from .nodes.lora_selector import NODE_DISPLAY_NAME_MAPPINGS as LORA_SELECTOR_NAMES

# Image utilities
from .nodes.image_resize import NODE_CLASS_MAPPINGS as IMAGE_RESIZE_MAPPINGS
from .nodes.image_resize import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_RESIZE_NAMES

# LLM nodes
from .nodes.llm_config import NODE_CLASS_MAPPINGS as LLM_CONFIG_MAPPINGS
from .nodes.llm_config import NODE_DISPLAY_NAME_MAPPINGS as LLM_CONFIG_NAMES
from .nodes.llm_prompt import NODE_CLASS_MAPPINGS as LLM_PROMPT_MAPPINGS
from .nodes.llm_prompt import NODE_DISPLAY_NAME_MAPPINGS as LLM_PROMPT_NAMES

# Image batch utilities
from .nodes.image_batch_utils import NODE_CLASS_MAPPINGS as IMAGE_BATCH_MAPPINGS
from .nodes.image_batch_utils import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_BATCH_NAMES

# ChatterBox TTS/VC nodes
from .nodes.chatterbox_nodes import NODE_CLASS_MAPPINGS as CHATTERBOX_MAPPINGS
from .nodes.chatterbox_nodes import NODE_DISPLAY_NAME_MAPPINGS as CHATTERBOX_NAMES

# Size selector
from .nodes.size_selector import NODE_CLASS_MAPPINGS as SIZE_SELECTOR_MAPPINGS
from .nodes.size_selector import NODE_DISPLAY_NAME_MAPPINGS as SIZE_SELECTOR_NAMES

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Loop nodes
NODE_CLASS_MAPPINGS.update(FOR_LOOP_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FOR_LOOP_NAMES)

# String list utilities
NODE_CLASS_MAPPINGS.update(STRING_LIST_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(STRING_LIST_NAMES)

# LoRA selector
NODE_CLASS_MAPPINGS.update(LORA_SELECTOR_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LORA_SELECTOR_NAMES)

# Image utilities
NODE_CLASS_MAPPINGS.update(IMAGE_RESIZE_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_RESIZE_NAMES)

# LLM nodes
NODE_CLASS_MAPPINGS.update(LLM_CONFIG_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LLM_CONFIG_NAMES)
NODE_CLASS_MAPPINGS.update(LLM_PROMPT_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LLM_PROMPT_NAMES)

# Image batch utilities
NODE_CLASS_MAPPINGS.update(IMAGE_BATCH_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_BATCH_NAMES)

# ChatterBox TTS/VC nodes
NODE_CLASS_MAPPINGS.update(CHATTERBOX_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CHATTERBOX_NAMES)

# Size selector
NODE_CLASS_MAPPINGS.update(SIZE_SELECTOR_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SIZE_SELECTOR_NAMES)

# Web directory for JavaScript extensions
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Version info
__version__ = "0.1.0"
__author__ = "Latent Astronaut"
