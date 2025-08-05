import runpod
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from io import BytesIO
import base64
import requests

print("Loading pipeline...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)
pipe.to("cuda")

def download_image(url):
    response = requests.get(url)
    image = load_image(BytesIO(response.content))
    return image

def handler(job):
    prompt = job["input"]["prompt"]
    image_url = job["input"]["image_url"]

    input_image = download_image(image_url)
    output = pipe(image=input_image, prompt=prompt, guidance_scale=2.5).images[0]

    buffered = BytesIO()
    output.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"image_base64": img_str}

runpod.serverless.start({"handler": handler})
