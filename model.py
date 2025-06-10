import os

MODEL_DIR = os.getenv("MODEL_DIR", "./models")

def model_inference(filename: str):
    # Check if filename exists in MODEL_DIR
    file_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(file_path):
        return {"message": "received picture"}
    else:
        return {"message": "picture not found"}
