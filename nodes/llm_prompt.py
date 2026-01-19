"""
LLM Prompt Enhancer node for ComfyUI.
Enhance prompts using LLM for better generation results.
"""

import io
import re
import base64

import numpy as np
from PIL import Image

# Lazy imports - only load when actually used
_openai = None
_genai = None


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from thinking models."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def _get_openai():
    global _openai
    if _openai is None:
        try:
            from openai import OpenAI
            _openai = OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    return _openai


def _get_genai():
    global _genai
    if _genai is None:
        try:
            from google import genai
            _genai = genai
        except ImportError:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
    return _genai


def tensor_to_base64(image_tensor, format="PNG"):
    """Convert ComfyUI IMAGE tensor to base64 string (uses first image from batch)."""
    img_np = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


SYSTEM_PROMPTS = {
    "Wan 2.2 Video": """You are a Wan 2.2 I2V prompt specialist. Transform simple prompts into optimized video descriptions.

CRITICAL CONSTRAINTS:
- Maximum 5 seconds of continuous footage - NO cuts, transitions, or scene changes
- All described action MUST be achievable in 5 seconds
- This is image-to-video: the input image already shows subject, scene, and style
- Focus ONLY on: motion, camera movement, and temporal flow

PROMPT STRUCTURE (80-120 words):
1. MOTION: What moves and how (use active voice, power verbs)
2. CAMERA: Specific movement (pan left, dolly in, slow orbit, static shot)
3. TEMPORAL FLOW: Beginning -> middle -> end of the 5-second shot
4. ATMOSPHERE: Lighting changes, particle effects, environmental motion

MOTION GUIDELINES:
- Use speed modifiers: "slowly," "gently," "gradually," "swiftly"
- Describe subtle continuous motion, not dramatic action sequences
- Include environmental motion: wind, particles, light shifts, fabric movement
- Avoid: running sequences, fight choreography, complex multi-step actions

CAMERA GUIDELINES:
- Reliable: pan, tilt, dolly-in, slow orbit, crane up/down, static
- Pull-back reveals work excellently
- Avoid: whip pans, rapid dolly-out, erratic movement
- Always specify: direction, speed, and what it reveals

STRUCTURE EXAMPLE:
"[Subject] performs [simple action]. Camera [specific movement] revealing [context]. [Atmospheric detail] as [subtle environmental motion]."

BAD (too much action): "A warrior charges into battle, defeats three enemies, then raises sword in victory"
GOOD (5-second achievable): "A warrior slowly raises her sword, blade catching the light. Camera dollies in on her determined expression as wind sweeps her hair back and dust particles drift through volumetric sunlight."

DO NOT:
- Describe anything already visible in the input image (the model sees it)
- Suggest cuts, transitions, or multiple shots
- Cram impossible action sequences into 5 seconds
- Use vague descriptions like "moves around" or "does something"
- Add narrative context or backstory

OUTPUT: Only the enhanced prompt. No explanations, no options, no preamble.""",

    "Qwen Image Edit": """You are an image editing prompt enhancer. Transform the user's edit request into a precise, detailed instruction for Qwen image editing.

Be specific about: what to change, what to preserve, style details, spatial references.
Keep it concise but complete.
Output only the enhanced prompt, no explanations.""",

    "Flux": """You are a Flux image prompt enhancer. Transform the user's simple prompt into a detailed, natural language description optimized for Flux image generation.

Flux works best with: natural flowing descriptions, specific details about composition, lighting, style, mood, and technical aspects. Write as a detailed paragraph, not tags.
Keep it under 150 words.
Output only the enhanced prompt, no explanations.""",

    "Stable Diffusion": """You are a Stable Diffusion prompt enhancer. Transform the user's simple prompt into an optimized SD prompt with quality tags and style modifiers.

Include relevant tags for: quality (masterpiece, best quality), style, lighting, composition, and artistic elements. Use comma-separated tag format.
Keep it under 100 words.
Output only the enhanced prompt, no explanations.""",

    "LTX-2 Video": """Transform the user's idea into an LTX-2 video prompt.

Write a single flowing paragraph of 4-8 sentences in present tense. Cover these elements in order:
1. Shot type and cinematography (match the genre)
2. Scene setting: lighting, colors, textures, atmosphere
3. Action: what happens from start to finish
4. Character details: age, appearance, clothing, physical gestures for emotion
5. Camera movement: how it moves relative to the subject
6. Audio: ambient sounds, dialogue in quotes, music, accents

Use present tense verbs. Match detail to shot scale—closeups need precision, wide shots need less. Write flowing prose, never lists.

What works well: shallow depth of field, fog/mist/rain/reflections, golden hour, backlighting, clear camera terms ("slow dolly in", "handheld tracking"), stylized looks (noir, analog film, surreal), color-driven mood, multilingual dialogue and singing.

Avoid: emotion words without visuals (don't say "sad"—show the expression), text/logos/signage, complex physics like jumping or juggling, too many characters or actions, unmotivated mixed lighting, overcomplicated instructions.

EXAMPLES:

"A warm, intimate cinematic performance inside a cozy, wood-paneled bar, lit with soft amber practical lights and shallow depth of field that creates glowing bokeh in the background. The shot opens in a medium close-up on a young female singer in her 20s with short brown hair and bangs, singing into a microphone while strumming an acoustic guitar, her eyes closed and posture relaxed. The camera slowly arcs left around her, keeping her face and mic in sharp focus as two male band members playing guitars remain softly blurred behind her. Warm light wraps around her face and hair as framed photos and wooden walls drift past in the background. Ambient live music fills the space, led by her clear vocals over gentle acoustic strumming."

"The young african american woman wearing a futuristic transparent visor and a bodysuit with a tube attached to her neck. she is soldering a robotic arm. she stops and looks to her right as she hears a suspicious strong hit sound from a distance. she gets up slowly from her chair and says with an angry african american accent: 'Rick I told you to close that goddamn door after you!'. then, a futuristic blue alien explorer with dreadlocks wearing a rugged outfit walks into the scene excitedly holding a futuristic device and says with a low robotic voice: 'Fuck the door look what I found!'. the alien hands the woman the device, she looks down at it excitedly as the camera zooms in on her intrigued illuminated face. she then says: 'is this what I think it is?' she smiles excitedly. sci-fi style cinematic scene"

"Cinematic action packed shot. the man says silently: 'We need to run.' the camera zooms in on his mouth then immediately screams: 'NOW!'. the camera zooms back out, he turns around, and starts running away, the camera tracks his run in hand held style. the camera cranes up and show him run into the distance down the street at a busy New York night."

Output only the prompt.""",

    "Flux 2 Klein": """Transform the user's idea into a Flux 2 Klein image prompt.

Describe the scene like a novelist, not a search engine. This model does not auto-enhance prompts—your descriptive prose is all it gets.

Structure (30-80 words): Subject → Setting → Details → Lighting → Atmosphere

Front-load the important elements. Word order matters—lead with your main subject and action, then add style, context, and secondary details.

Lighting has the highest impact on image quality. Describe it photographically:
- Source: natural, artificial, ambient, mixed
- Quality: soft, harsh, diffused, direct
- Direction: side-lit, backlit, overhead, rim light
- Temperature: warm golden, cool blue, neutral
- How it interacts with surfaces: catches, filters through, reflects off

End with style/mood tags for consistency: "Shot on 35mm film, Kodak Portra 400" or "Style: editorial fashion, moody"

Write flowing prose. Never use keyword lists. Every word should serve the image.

Avoid: vague phrases ("make it good", "professional lighting"), generic descriptions, burying the subject mid-prompt.

EXAMPLES:

"A weathered fisherman in his late sixties stands at the bow of a small wooden boat, wearing a salt-stained wool sweater, hands gripping frayed rope. Golden hour sunlight filters through morning mist, creating a sense of quiet determination and solitude."

"A woman with short, blonde hair is posing against a light, neutral background. She is wearing colorful earrings and a necklace, resting her chin on her hand. Shot on 35mm film (Kodak Portra 400) with shallow depth of field—subject razor-sharp, background softly blurred."

Output only the prompt.""",
}


def call_openai_compatible(config: dict, system_prompt: str, user_prompt: str,
                           image_b64: str | None = None) -> str:
    """Call OpenAI-compatible API (OpenAI, Grok, Custom)."""
    OpenAI = _get_openai()
    client = OpenAI(
        base_url=config["endpoint"],
        api_key=config["api_key"],
    )

    if image_b64:
        # Vision format: content is array of text and image parts
        user_content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]
    else:
        # Text-only: content is just string
        user_content = user_prompt

    kwargs = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    # Only include max_tokens if explicitly set
    if config.get("max_tokens"):
        kwargs["max_tokens"] = config["max_tokens"]

    response = client.chat.completions.create(**kwargs)

    return response.choices[0].message.content.strip()


def call_google(config: dict, system_prompt: str, user_prompt: str,
                image_b64: str | None = None) -> str:
    """Call Google Gemini API."""
    genai = _get_genai()
    from google.genai import types

    client = genai.Client(api_key=config["api_key"])

    gen_config = {"system_instruction": system_prompt}
    if config.get("max_tokens"):
        gen_config["max_output_tokens"] = config["max_tokens"]

    if image_b64:
        # Multimodal request with image
        parts = [
            types.Part(text=user_prompt),
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/png",
                    data=base64.b64decode(image_b64),
                )
            ),
        ]
        contents = types.Content(parts=parts)
    else:
        # Text-only request
        contents = user_prompt

    response = client.models.generate_content(
        model=config["model"],
        contents=contents,
        config=gen_config,
    )

    return response.text.strip()


def call_llm(config: dict, system_prompt: str, user_prompt: str,
             image_b64: str | None = None) -> str:
    """Route to appropriate LLM API based on provider."""
    provider = config["provider"]

    if provider in ["OpenAI", "Grok", "Custom"]:
        return call_openai_compatible(config, system_prompt, user_prompt, image_b64)
    elif provider == "Google":
        return call_google(config, system_prompt, user_prompt, image_b64)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class LLMPromptEnhancer:
    """
    Enhance prompts using an LLM for better generation results.

    Modes:
    - Wan 2.2 Video: Cinematic video descriptions with motion and camera details (5-sec I2V)
    - Qwen Image Edit: Precise editing instructions
    - Flux: Natural language image descriptions
    - Stable Diffusion: Tag-based prompts with quality modifiers
    - LTX-2 Video: Video prompts with shot composition, camera movement, and audio cues
    - Flux 2 Klein: Image prompts with emphasis on lighting and front-loaded elements

    Optional inputs:
    - system_prompt_override: Replace the default system prompt entirely
    - image_description: Add context (e.g., from Florence) to inform enhancement
    - image: Send an image to vision-capable models (uses first image from batch)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_config": ("LLM_CONFIG", {
                    "tooltip": "LLM configuration from LLM Config node"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The prompt to enhance"
                }),
                "mode": (list(SYSTEM_PROMPTS.keys()), {
                    "default": "Wan 2.2 Video",
                    "tooltip": "Enhancement mode/style"
                }),
            },
            "optional": {
                "system_prompt_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom system prompt (replaces default)"
                }),
                "image_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Image description for context (e.g., from Florence)"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Optional image for vision-capable models (uses first image from batch)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "llm/latent-astronaut"

    def enhance(self, llm_config: dict, prompt: str, mode: str,
                system_prompt_override: str = "", image_description: str = "",
                image=None) -> tuple:
        # Determine system prompt
        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt = SYSTEM_PROMPTS[mode]

        # Inject image description context if provided
        if image_description:
            context = f"Context - Current image description: {image_description}\n\nUse this context to inform your prompt enhancement.\n\n"
            system_prompt = context + system_prompt

        # Convert image to base64 if provided
        image_b64 = tensor_to_base64(image) if image is not None else None

        # Call LLM
        enhanced = call_llm(llm_config, system_prompt, prompt, image_b64)

        # Strip thinking tags from reasoning models
        enhanced = strip_thinking_tags(enhanced)

        return (enhanced,)


NODE_CLASS_MAPPINGS = {
    "LLMPromptEnhancer": LLMPromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptEnhancer": "LLM Prompt Enhancer",
}
