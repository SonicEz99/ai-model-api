from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import easyocr

app = FastAPI()

# Load YOLO model for number plate detection
model = YOLO("./best.pt")  # Update with your model path

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

async def detect_number_plate(img):
    """
    Detect number plates in the image and extract text using OCR
    """
    try:
        # Run YOLO detection
        results = model(img)
        plates = []
        
        for r in results:
            if r.boxes:  # if detected boxes
                for box in r.boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop the plate region from the image
                    plate_img = img[y1:y2, x1:x2]
                    
                    # Skip if cropped image is too small
                    if plate_img.shape[0] < 10 or plate_img.shape[1] < 10:
                        continue
                    
                    # Perform OCR on cropped plate image
                    ocr_result = reader.readtext(plate_img)
                    
                    # Extract text from OCR results
                    if ocr_result:
                        text = "".join([res[1] for res in ocr_result if res[2] > 0.5])  # Confidence threshold
                        if text.strip():  # Only add non-empty text
                            plates.append(text.strip())
        
        return plates[0] if plates else ""
    
    except Exception as e:
        print(f"❌ Error in number plate detection: {e}")
        return ""

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connection accepted")
    
    try:
        while True:
            # Receive video frame data from client
            data = await websocket.receive_bytes()
            
            # Decode the received frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                print("📷 Frame received and decoded successfully")
                
                # Detect number plates and extract text
                plate_text = await detect_number_plate(frame)
                
                if plate_text:
                    print(f"🔍 Detected number plate: {plate_text}")
                    await websocket.send_json({
                        "numberPlate": plate_text,
                        "status": "detected"
                    })
                else:
                    print("⚠️ No number plate detected")
                    await websocket.send_json({
                        "numberPlate": "No number plate detected",
                        "status": "not_detected"
                    })
            else:
                print("⚠️ Failed to decode frame")
                await websocket.send_json({
                    "numberPlate": "Frame decode error",
                    "status": "error"
                })
                
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.send_json({
            "numberPlate": "Processing error",
            "status": "error"
        })
    finally:
        print("🔌 WebSocket connection closed")

# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Number plate detection service is running"}

# Optional: Add CORS middleware if needed for frontend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)