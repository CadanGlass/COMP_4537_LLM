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
    Depends,
    Header
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import jwt
from jwt.exceptions import PyJWTError
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch


# Load environment variables from .env file
load_dotenv()

# Access the SECRET_KEY and ALGORITHM from environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

app = FastAPI(title="Image-to-Text Generator")

# CORS configuration
origins = [
    "https://cadan.xyz",
    "http://localhost:3000",
    "http://localhost:5173",  # Added frontend's local development port
    # Add other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# JWT Configuration
class TokenData(BaseModel):
    username: str
    role: str


# Dependency to extract the token from the Authorization header
async def get_token(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    parts = authorization.split()
    if parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization header must start with Bearer")
    elif len(parts) == 1:
        raise HTTPException(status_code=401, detail="Token not found")
    elif len(parts) > 2:
        raise HTTPException(status_code=401, detail="Authorization header must be Bearer + \\s + token")
    token = parts[1]
    return token


def verify_token(token: str = Depends(get_token)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        role: str = payload.get("role")
        if username is None or role is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        token_data = TokenData(username=username, role=role)
        return token_data
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Load the smaller image captioning model and processor
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

# Define generation parameters for efficiency
max_length = 16
num_beams = 4


@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...), token_data: TokenData = Depends(verify_token)):
    """
    Endpoint to generate a caption for an uploaded image.
    Requires a valid JWT token.
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
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Generate the caption
        output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return JSONResponse(content={"caption": caption})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating caption") from e


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Image-to-Text Generator API. Use the /generate-caption/ endpoint to generate captions for your images."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
