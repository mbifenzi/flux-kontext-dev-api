from fastapi import FastAPI, UploadFile, Form
from diffusers import FluxKontextPipeline
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import io

app = FastAPI()

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

@app.post("/edit")
async def edit_image(file: UploadFile, prompt: str = Form(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = pipe(image=img, prompt=prompt, guidance_scale=2.5).images[0]
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
