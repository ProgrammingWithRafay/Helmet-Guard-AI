import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gradio as gr
import spaces
from app.main import app as fastapi_app

@spaces.GPU
def api_status():
    return "HelmetGuard AI API is successfully running on Hugging Face!"

# Create a tiny placeholder Gradio interface so Hugging Face is happy
demo = gr.Interface(
    fn=api_status, 
    inputs=None, 
    outputs="text",
    title="HelmetGuard AI API"
)

# Mount our massive FastAPI backend directly into the Gradio app!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
