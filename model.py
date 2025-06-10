import os

MODEL_DIR = os.getenv("MODEL_DIR", "./models")

def model_inference(filename: str):
    # Build full path
    file_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(file_path):
        result = {"message": "received picture"}
        try:
            os.remove(file_path)  # Delete the file after processing
        except Exception as e:
            result["warning"] = f"Could not delete file: {e}"
        return result
    else:
        return {"message": "picture not found"}
