import torch
from diffusers import FluxKontextPipeline
from PIL import Image
import base64
import io

# Load the model once into GPU memory
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.float16  # Use float16 to ensure compatibility with A10 GPUs
).to("cuda")

def handler(event):
    try:
        inputs = event.get("input", {})
        prompt = inputs.get("prompt")
        image_base64 = inputs.get("image")

        if not prompt or not image_base64:
            return {"error": "Both 'prompt' and base64 'image' are required."}

        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        result = pipe(image=image, prompt=prompt, guidance_scale=2.5).images[0]

        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        output_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"output": output_base64}

    except Exception as e:
        return {"error": str(e)}
