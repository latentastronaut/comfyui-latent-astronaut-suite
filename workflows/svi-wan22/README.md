# SVI Looping Workflow v1

---

## Resources and Models

| Resource | Link |
|----------|------|
| SVI main page | https://github.com/vita-epfl/Stable-Video-Infinity |
| KJnodes | https://github.com/kijai/ComfyUI-KJNodes |
| Wan fp8 models | https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main/I2V |
| Lightx2v models | https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/tree/main |
| SVI 2.0 Pro models | https://huggingface.co/Kijai/WanVideo_comfy/tree/main/LoRAs/Stable-Video-Infinity/v2.0 |
| CLIP | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/text_encoders |
| VAE | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/vae |
| GGUFs | https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/tree/main |
| AI Search Video | https://www.youtube.com/watch?v=-3DVJu72VhE |
| Prompt GPT Helper | https://chatgpt.com/g/g-6887849e21b8819183e20c1dc6bcf353-wan-2-2-prompt-generator |

---

## Info

**First Gen Prompt:** This is the first 5 seconds of your video

**Extend Prompts:** The workflow will go through the list of prompts, one per line, and feed it into the subsequent Extend Video generation

**Loras By Index:** For each video segment after the first one, you can choose a lora. These are loaded by index, so for the first extend segment, `lora_0` will be used.

---

## Notes

SVI seems to kill prompt adherence of Wan 2.2 I2V IMO, best when used with LORAs. It also seems to make the slow motion problem worse.
