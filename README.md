# Text-To-Video
Generates short videos from text prompts using Hugging Face’s ModelScopeT2V. A simple Python project exploring AI-powered text-to-video synthesis for creative content generation.
🚀 Example: Generate 64-frame Video using ModelScopeT2V with Optimizations
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# Load model with fp16 precision
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
)

# Optimize memory usage
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Generate a 64-frame video
prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=64).frames[0]
video_path = export_to_video(video_frames)
video_path
