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
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

class TextToVideoGenerator:
    def __init__(self, model_name="damo-vilab/text-to-video-ms-1.7b", device="cuda"):
        """
        Initializes the pipeline with the specified model and device.
        """
        print("🔄 Loading the model...")
        self.pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.device = device
        self.pipe = self.pipe.to(self.device)
        print("✅ Model loaded successfully.")

    def generate_video(self, prompt: str, num_frames: int = 16, steps: int = 25) -> str:
        """
        Generates video frames from a text prompt and exports the video.

        Args:
            prompt (str): The text description to convert to video.
            num_frames (int): Number of frames in the output video.
            steps (int): Number of inference steps.

        Returns:
            str: Path to the saved video file.
        """
        print(f"🎬 Generating video for prompt: '{prompt}'")
        result = self.pipe(prompt, num_inference_steps=steps, num_frames=num_frames)
        video_frames = result.frames[0]
        video_path = export_to_video(video_frames)
        print(f"📁 Video saved at: {video_path}")
        return video_path

# Usage
if __name__ == "__main__":
    prompt = "A panda dancing on the moon"
    
    # Create the generator object
    generator = TextToVideoGenerator()

    # Generate video
    generator.generate_video(prompt=prompt, num_frames=24, steps=30)
