from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import uuid

app = FastAPI()

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}_{image.filename}"
        file_path = os.path.join(temp_dir, filename)

        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)

        # TODO: Replace this with actual model inference
        result = {"prediction": "cat", "confidence": 0.92}

        os.remove(file_path)  # Clean up
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
