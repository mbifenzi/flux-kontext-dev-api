import subprocess
import uuid
import base64
from PIL import Image
from io import BytesIO
import os

def save_base64_image(b64_string, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_string))

def encode_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def handler(event):
    prompt = event.get("prompt")
    image_b64 = event.get("image")

    if not prompt or not image_b64:
        return {"error": "Missing 'prompt' or 'image' (base64)."}

    input_path = f"/tmp/input-{uuid.uuid4()}.png"
    output_dir = "/tmp/out"
    output_path = f"{output_dir}/flux_output.png"

    os.makedirs(output_dir, exist_ok=True)
    save_base64_image(image_b64, input_path)

    try:
        subprocess.run([
            "python3", "-m", "flux", "kontext",
            "--prompt", prompt,
            "--image", input_path,
            "--output_dir", output_dir
        ], check=True)

        result = encode_base64_image(output_path)
        return {"output": result}
    except subprocess.CalledProcessError as e:
        return {"error": f"FLUX crashed: {e}"}
    except Exception as e:
        return {"error": str(e)}
