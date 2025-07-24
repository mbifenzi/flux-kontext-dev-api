import torch
from diffusers import FluxKontextPipeline
from PIL import Image
import base64
import io

# Load the model once
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# RunPod serverless handler
def handler(event):
    try:
        inputs = event.get("input", {})
        prompt = inputs.get("prompt")
        image_base64 = inputs.get("image")

        if not prompt or not image_base64:
            return {"error": "Both 'prompt' and base64 'image' are required."}

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Run the model
        result = pipe(image=image, prompt=prompt, guidance_scale=2.5).images[0]

        # Encode the output
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        output_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"output": output_base64}

    except Exception as e:
        return {"error": str(e)}
