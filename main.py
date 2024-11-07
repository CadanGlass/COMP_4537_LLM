# main.py

# Standard Library Imports
import io
import os
from typing import Optional

# Third-Party Imports
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Header,
    Form
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Image-to-Text Generator")

# CORS configuration
origins = [
    "https://4537llm.online",
    "https://www.4537llm.online",
    "https://comp4537-term-project.netlify.app",
    "https://spontaneous-vacherin-eda866.netlify.app",
    "http://localhost:3000",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
    ],
    expose_headers=["*"],
)

# Load the image captioning model
try:
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model '{model_name}' loaded on {device}")
except Exception as e:
    print("Error loading model:", e)

max_length = 16
num_beams = 4

# Routes

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image-to-Text Generator API."}


@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Invalid image file") from e

    try:
        pixel_values = feature_extractor(
            images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(
            pixel_values, max_length=max_length, num_beams=num_beams)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return JSONResponse(content={"caption": caption})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error generating caption") from e

# Update the port to 8001 to match your frontend
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
