"""
LLM Config node for ComfyUI.
Configure LLM provider settings for use with other LLM nodes.
"""

import os


PROVIDER_DEFAULTS = {
    "OpenAI": {
        "endpoint": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "env_var": "OPENAI_API_KEY",
    },
    "Grok": {
        "endpoint": "https://api.x.ai/v1",
        "model": "grok-2",
        "env_var": "XAI_API_KEY",
    },
    "Google": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-1.5-flash",
        "env_var": "GOOGLE_API_KEY",
    },
    "Custom": {
        "endpoint": "",
        "model": "",
        "env_var": "",
    },
}


def resolve_api_key(key_input: str, provider: str) -> str:
    """
    Resolve API key from input or environment variable.

    Supports:
    - Direct key input
    - $ENV.VAR_NAME syntax
    - Empty string falls back to provider's default env var
    """
    if not key_input:
        # Fall back to provider's default env var
        env_var = PROVIDER_DEFAULTS.get(provider, {}).get("env_var", "")
        if env_var:
            return os.environ.get(env_var, "")
        return ""

    if key_input.startswith("$ENV."):
        env_var = key_input[5:]
        return os.environ.get(env_var, "")

    return key_input


class LLMConfig:
    """
    Configure LLM provider settings.

    Supports OpenAI, Grok (xAI), Google (Gemini), and custom OpenAI-compatible endpoints.

    API Key options:
    - Leave empty to use default environment variable for provider
    - Enter key directly (warning: may leak to workflow metadata)
    - Use $ENV.VAR_NAME to reference an environment variable
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (list(PROVIDER_DEFAULTS.keys()), {
                    "default": "OpenAI",
                    "tooltip": "LLM provider to use"
                }),
                "model": ("STRING", {
                    "default": "gpt-4o",
                    "tooltip": "Model name (e.g., gpt-4o, grok-2, gemini-1.5-flash)"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key, $ENV.VAR_NAME, or leave empty for default env var"
                }),
            },
            "optional": {
                "custom_endpoint": ("STRING", {
                    "default": "",
                    "tooltip": "Custom OpenAI-compatible endpoint URL (for Custom provider)"
                }),
                "max_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128000,
                    "tooltip": "Max tokens in response (0 = let API decide, required for some models)"
                }),
            },
        }

    RETURN_TYPES = ("LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "create_config"
    CATEGORY = "llm/latent-astronaut"

    def create_config(self, provider: str, model: str, api_key: str,
                      custom_endpoint: str = "", max_tokens: int = 0) -> tuple:
        # Resolve the API key
        resolved_key = resolve_api_key(api_key, provider)

        # Determine endpoint
        if provider == "Custom":
            endpoint = custom_endpoint
        else:
            endpoint = PROVIDER_DEFAULTS[provider]["endpoint"]

        # Use default model if not specified
        if not model:
            model = PROVIDER_DEFAULTS[provider]["model"]

        config = {
            "provider": provider,
            "model": model,
            "api_key": resolved_key,
            "endpoint": endpoint,
            "max_tokens": max_tokens if max_tokens > 0 else None,
        }

        return (config,)


NODE_CLASS_MAPPINGS = {
    "LLMConfig": LLMConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMConfig": "LLM Config",
}
