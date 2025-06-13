# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from model import model_inference

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ใช้ใน dev เท่านั้น
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    save_path = os.path.join(MODEL_DIR, image.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = model_inference(image.filename)
    return JSONResponse(content=result)
