import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from model import model_inference

load_dotenv()  # Load env variables from .env

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
API_KEY = os.getenv("API_KEY")

app = FastAPI(debug=DEBUG)

def verify_api_key(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/detect")
async def detect(image: UploadFile = File(...), x_api_key: str | None = Header(None)):
    verify_api_key(x_api_key)

    try:
        # We just want to check if the file name exists inside /models folder,
        # so no need to save the file permanently.
        # But you can optionally save if needed.

        filename = image.filename

        result = model_inference(filename)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
