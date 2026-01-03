"""
String List helper node for ComfyUI.
Converts multi-line text to a list of strings with wildcard expansion.
"""

import random
import re


def expand_wildcards(text: str, seed: int = None) -> str:
    """
    Expand wildcard syntax {option1|option2|option3} by randomly choosing one.

    Args:
        text: String potentially containing {a|b|c} patterns
        seed: Optional seed for reproducibility

    Returns:
        String with all wildcards expanded
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Pattern matches {option1|option2|...}
    pattern = r'\{([^{}]+)\}'

    def replace_match(match):
        options = match.group(1).split('|')
        return rng.choice(options).strip()

    # Keep expanding until no more wildcards (handles nested cases)
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = re.sub(pattern, replace_match, text)

    return text


class StringListFromText:
    """
    Convert multi-line text to a list of strings.
    Supports wildcard syntax: {red|blue|green} randomly picks one option.
    Empty lines are skipped.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "a {red|blue|green} cat\na {happy|sad} dog\na flying bird",
                    "tooltip": "One item per line. Use {a|b|c} for random selection."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for wildcard randomization (0 = random each time)"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("strings", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "parse"
    CATEGORY = "utils/latent-astronaut"

    def parse(self, text, seed):
        # Use None for seed=0 to get truly random behavior
        actual_seed = seed if seed != 0 else None

        lines = text.strip().split('\n')
        result = []

        for i, line in enumerate(lines):
            line = line.strip()
            if line:  # Skip empty lines
                # Use seed + line index for different randomization per line
                line_seed = (actual_seed + i) if actual_seed is not None else None
                expanded = expand_wildcards(line, line_seed)
                result.append(expanded)

        return (result, len(result))


class StringListCombine:
    """
    Combine multiple string lists into one.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_1": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "list_2": ("STRING", {"forceInput": True}),
                "list_3": ("STRING", {"forceInput": True}),
                "list_4": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("strings",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "combine"
    CATEGORY = "utils/latent-astronaut"

    def combine(self, list_1, list_2=None, list_3=None, list_4=None):
        result = list(list_1) if list_1 else []
        if list_2:
            result.extend(list_2)
        if list_3:
            result.extend(list_3)
        if list_4:
            result.extend(list_4)
        return (result,)


class StringListIndex:
    """
    Get a single string from a list by index.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strings": ("STRING", {"forceInput": True}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Index of string to get (0-based)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    INPUT_IS_LIST = True
    FUNCTION = "get_index"
    CATEGORY = "utils/latent-astronaut"

    def get_index(self, strings, index):
        # Index comes as list too when INPUT_IS_LIST=True
        idx = index[0] if isinstance(index, list) else index
        strings_list = strings if isinstance(strings, list) else [strings]

        if idx >= len(strings_list):
            idx = len(strings_list) - 1

        return (strings_list[idx],)


NODE_CLASS_MAPPINGS = {
    "StringListFromText": StringListFromText,
    "StringListCombine": StringListCombine,
    "StringListIndex": StringListIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringListFromText": "String List from Text",
    "StringListCombine": "String List Combine",
    "StringListIndex": "String List Index",
}
