from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = FastAPI()

# 1. Configura dispositivo y carga el modelo + transforms desde Torch Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo DPT Swin-Tiny 256 v3.1:
midas = torch.hub.load(
    "isl-org/MiDaS",            # repo oficial
    "dpt_swin2_tiny_256",       # nombre del modelo
    pretrained=True,
    trust_repo=True             # evita validación de GitHub API (rate limit)
)
midas.to(device).eval()

# Transforms oficiales
transforms = torch.hub.load(
    "isl-org/MiDaS",
    "transforms",
    trust_repo=True             # idem para los transforms
)
transform = transforms.dpt_transform  # para redes DPT

@app.post("/visualizar")
async def visualizar(pared: UploadFile = File(...), cuadro: UploadFile = File(...)):
    try:
        # Leer imágenes subidas
        pared_bytes = await pared.read()
        cuadro_bytes = await cuadro.read()
        img_pared = Image.open(io.BytesIO(pared_bytes)).convert("RGB")
        img_cuadro = Image.open(io.BytesIO(cuadro_bytes)).convert("RGBA")

        # 2. Preprocesado + forward al modelo
        inp = transform(img_pared).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = midas(inp).squeeze().cpu().numpy()

        # 3. Post-procesado del mapa de profundidad
        h, w = np.array(img_pared).shape[:2]
        pred_resized = cv2.resize(pred, (w, h))
        pred_norm = cv2.normalize(pred_resized, None, 0, 1, cv2.NORM_MINMAX)

        # 4. Superponer el cuadro sobre la pared
        base = np.array(img_pared).astype(np.float32)
        cuadro_np = np.array(img_cuadro).astype(np.float32)
        alpha = cuadro_np[..., 3:] / 255.0

        y0 = h//2 - cuadro_np.shape[0]//2
        x0 = w//2 - cuadro_np.shape[1]//2

        # Mezcla canal por canal
        for c in range(3):
            base[y0:y0+cuadro_np.shape[0], x0:x0+cuadro_np.shape[1], c] = (
                cuadro_np[..., c] * alpha[..., 0]
                + base[y0:y0+cuadro_np.shape[0], x0:x0+cuadro_np.shape[1], c] * (1 - alpha[..., 0])
            )

        # 5. Devolver PNG en base64
        final_img = Image.fromarray(base.astype(np.uint8))
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return JSONResponse(content={"base64": b64})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "MiDaS backend funcionando"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
