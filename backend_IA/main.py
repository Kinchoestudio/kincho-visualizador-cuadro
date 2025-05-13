# backend_IA/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import base64
import numpy as np
import cv2
import os

# Asegúrate de haber añadido en requirements.txt:
#   git+https://github.com/isl-org/MiDaS.git@v3_1
#   timm

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Descarga manual de pesos v3.1
model_path = "midas/weights/dpt_swin2_tiny_256.pt"
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.system(
        "wget "
        "https://github.com/isl-org/MiDaS/releases/download/v3_1/"
        "dpt_swin2_tiny_256.pt -O " + model_path
    )

model = DPTDepthModel(
    path=model_path,
    backbone="dpt_swin2_tiny_256",  # se strippea automáticamente en dpt_depth.py
    non_negative=True
)
model.to(device).eval()

# Transforms oficiales MiDaS
transform = Resize(
    256, 256,
    resize_target=None,
    keep_aspect_ratio=True,
    ensure_multiple_of=32,
    resize_method="minimal",
    image_interpolation_method=cv2.INTER_CUBIC
)
normalize = NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
prepare = PrepareForNet()

@app.post("/visualizar")
async def visualizar(pared: UploadFile = File(...), cuadro: UploadFile = File(...)):
    try:
        img_pared = Image.open(io.BytesIO(await pared.read())).convert("RGB")
        img_cuadro = Image.open(io.BytesIO(await cuadro.read())).convert("RGBA")

        img_np = np.array(img_pared)
        inp = prepare(normalize(transform({"image": img_np})["image"]))
        sample = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(sample).cpu().numpy().squeeze()

        h, w = img_np.shape[:2]
        pred = cv2.normalize(
            cv2.resize(pred, (w, h)),
            None, 0, 1, cv2.NORM_MINMAX
        )

        base = img_np.astype(np.float32)
        cu = np.array(img_cuadro).astype(np.float32)
        alpha = cu[..., 3:] / 255.0
        y0, x0 = (h - cu.shape[0]) // 2, (w - cu.shape[1]) // 2
        for c in range(3):
            base[y0:y0+cu.shape[0], x0:x0+cu.shape[1], c] = (
                cu[..., c] * alpha[..., 0] +
                base[y0:y0+cu.shape[0], x0:x0+cu.shape[1], c] * (1 - alpha[..., 0])
            )

        out = Image.fromarray(base.astype(np.uint8))
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return JSONResponse({"base64": base64.b64encode(buf.getvalue()).decode()})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "MiDaS backend funcionando"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
