import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from io import BytesIO
import base64
from PIL import Image

# Load model once
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.float16
).to("cuda")

def decode_base64_image(b64_string):
    return Image.open(BytesIO(base64.b64decode(b64_string)))

def encode_base64_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handler(event):
    prompt = event.get("prompt", "")
    image_b64 = event.get("image", "")

    if not prompt or not image_b64:
        return {"error": "Both 'prompt' and 'image' (base64) are required."}

    try:
        input_image = decode_base64_image(image_b64)

        edited = pipe(
            image=input_image,
            prompt=prompt,
            guidance_scale=2.5
        ).images[0]

        return {
            "output": encode_base64_image(edited)
        }
    except Exception as e:
        return {"error": str(e)}
