from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Backend de visualización activo correctamente"}
# Código principal FastAPI (simulado para ejemplo)
