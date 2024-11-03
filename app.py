# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

app = FastAPI(title="Image-to-Text Generator")

# Load the BLIP model and processor
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}")
except Exception as e:
    print("Error loading model:", e)

@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    """
    Endpoint to generate a caption for an uploaded image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file") from e

    try:
        # Prepare the image for the model
        inputs = processor(image, return_tensors="pt").to(device)

        # Generate the caption
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return JSONResponse(content={"caption": caption})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating caption") from e

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image-to-Text Generator API. Use the /generate-caption/ endpoint to generate captions for your images."}
