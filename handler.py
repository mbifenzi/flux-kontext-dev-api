import subprocess
import base64
import uuid
from PIL import Image
from io import BytesIO

def save_base64_image(b64_string, path):
    image_data = base64.b64decode(b64_string)
    with open(path, "wb") as f:
        f.write(image_data)

def encode_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def handler(event):
    prompt = event.get("prompt")
    image_b64 = event.get("image")

    if not prompt or not image_b64:
        return {"error": "Missing prompt or image."}

    input_path = f"/tmp/input-{uuid.uuid4()}.png"
    output_path = "/tmp/out/flux_output.png"

    save_base64_image(image_b64, input_path)

    try:
        subprocess.run([
            "python3", "-m", "flux", "kontext",
            "--prompt", prompt,
            "--image", input_path,
            "--output_dir", "/tmp/out"
        ], check=True)

        output_b64 = encode_base64_image(output_path)
        return {"output": output_b64}
    except subprocess.CalledProcessError as e:
        return {"error": f"FLUX crashed: {e}"}
    except Exception as ex:
        return {"error": str(ex)}
