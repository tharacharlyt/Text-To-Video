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

    import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# Load the model from Hugging Face
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b", 
    torch_dtype=torch.float16, 
    variant="fp16"
)

# Move to GPU
pipe = pipe.to("cuda")

# Define a simple prompt
prompt = "A cute dog running in the park"

# Generate video frames from the prompt
output = pipe(prompt)
video_frames = output.frames[0]

# Export the frames to a video file
video_path = export_to_video(video_frames)

# Print the path to the generated video
print(f"Video saved at: {video_path}")
Install Required Packages
Before running this code, install dependencies using:

pip install diffusers transformers accelerate torch torchvision imageio[ffmpeg]
🖥️ Requirements
GPU is required (e.g., NVIDIA with at least 12 GB VRAM).
Python 3.8+
📝 An input text box for the prompt
🎥 An output area for the generated video
🔠 A dropdown to select font style for display
✅ Install Streamlit First:
pip install streamlit
📄 app.py — Streamlit Web App
import streamlit as st
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import os

# Set up page config
st.set_page_config(page_title="Text-to-Video Generator", layout="centered")

# Font options
font_options = ["Arial", "Courier New", "Georgia", "Verdana", "Times New Roman"]
font_choice = st.selectbox("Choose a font style", font_options)

# Apply custom font using Streamlit HTML
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: '{font_choice}', sans-serif;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🎬 Text-to-Video Generator")

# Input box for prompt
prompt = st.text_input("Enter your video prompt", value="A robot walking in the forest")

# Button to generate
if st.button("Generate Video"):
    with st.spinner("Generating video... Please wait ⏳"):
        # Load the pipeline (runs only once)
        @st.cache_resource
        def load_pipeline():
            pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b", 
                torch_dtype=torch.float16, 
                variant="fp16"
            )
            return pipe.to("cuda")

        pipe = load_pipeline()

        # Generate video
        output = pipe(prompt, num_inference_steps=25)
        video_frames = output.frames[0]
        video_path = export_to_video(video_frames)

        # Display video
        st.success("✅ Video generated successfully!")
        st.video(video_path)

# Optional footer
st.markdown("---")
st.caption("Built with ❤️ using Hugging Face Diffusers and Streamlit")
🔧 Run the App
In your terminal:

streamlit run app.py
💡 Features
Font selection applies to all text dynamically.
Input prompt → Hugging Face model → output video displayed.
Easy to customize with new fonts, styling, and model settings.

