from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose
from PIL import Image
import base64
import io
import os
import sys

app = FastAPI()

# Clonar MiDaS si no existe
if not os.path.exists("MiDaS"):
    os.system("git clone https://github.com/isl-org/MiDaS.git")
sys.path.append("MiDaS")

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# Modelo ligero compatible con Render Free
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "MiDaS/weights/dpt_levit_224.pt"
if not os.path.exists(model_path):
    os.makedirs("MiDaS/weights", exist_ok=True)
    os.system(f"wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt -O {model_path}")

model = DPTDepthModel(
    path=model_path,
    backbone="dpt_levit_224",
    non_negative=True,
)
model.eval()
model.to(device)

transform = Compose([
    Resize(224, 224, resize_target=None, keep_aspect_ratio=True,
           ensure_multiple_of=32, resize_method="minimal",
           image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    PrepareForNet()
])

@app.post("/visualizar")
async def visualizar(pared: UploadFile = File(...), cuadro: UploadFile = File(...)):
    try:
        pared_bytes = await pared.read()
        cuadro_bytes = await cuadro.read()

        img_pared = Image.open(io.BytesIO(pared_bytes)).convert("RGB")
        img_cuadro = Image.open(io.BytesIO(cuadro_bytes)).convert("RGBA")

        img = np.array(img_pared)
        img_input = transform({"image": img})["image"]
        sample = torch.from_numpy(img_input).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model.forward(sample)
            prediction = prediction.squeeze().cpu().numpy()

        prediction_resized = cv2.resize(prediction, (img.shape[1], img.shape[0]))
        prediction_resized = cv2.normalize(prediction_resized, None, 0, 1, cv2.NORM_MINMAX)

        h, w = img.shape[:2]
        x_offset = w // 2 - img_cuadro.width // 2
        y_offset = h // 2 - img_cuadro.height // 2

        base = img.copy()
        cuadro_np = np.array(img_cuadro)
        for c in range(3):
            base[y_offset:y_offset+cuadro_np.shape[0], x_offset:x_offset+cuadro_np.shape[1], c] = \
                cuadro_np[..., c] * (cuadro_np[..., 3] / 255.0) + \
                base[y_offset:y_offset+cuadro_np.shape[0], x_offset:x_offset+cuadro_np.shape[1], c] * (1.0 - cuadro_np[..., 3] / 255.0)

        final_image = Image.fromarray(base)
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return JSONResponse(content={"base64": encoded})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "MiDaS backend funcionando con modelo liviano"}
