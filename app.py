from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
from fastapi.middleware.cors import CORSMiddleware
import jwt
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import Optional
from fastapi import Header
from jwt import PyJWTError  # Correct import for PyJWTError
# Load environment variables from .env file
load_dotenv()
# Access the SECRET_KEY and ALGORITHM from environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
app = FastAPI(title="Image-to-Text Generator")
# CORS configuration
origins = [
    "https://4537llm.online",                        # Your HTTPS domain
    "https://www.4537llm.online",                    # WWW subdomain
    "https://comp4537-term-project.netlify.app",     # Hosted frontend URL
    "https://spontaneous-vacherin-eda866.netlify.app",  # New frontend URL
    "http://localhost:3000",                         # Local development
    "http://localhost:5173"    
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allow specified origins
    allow_credentials=True,
    allow_methods=["*"],              # Allow all HTTP methods
    allow_headers=["*"],              # Allow all headers
)
# JWT Configuration
class TokenData(BaseModel):
    username: str
    role: str
# Dependency to extract the token from the Authorization header
async def get_token(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(
            status_code=401, detail="Authorization header missing")
    parts = authorization.split()
    if parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401, detail="Authorization header must start with Bearer")
    elif len(parts) == 1:
        raise HTTPException(status_code=401, detail="Token not found")
    elif len(parts) > 2:
        raise HTTPException(
            status_code=401, detail="Authorization header must be Bearer + \\s + token")
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
# Load the BLIP model and processor
try:
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}")
except Exception as e:
    print("Error loading model:", e)
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
        raise HTTPException(
            status_code=400, detail="Invalid image file") from e
    try:
        # Prepare the image for the model
        inputs = processor(image, return_tensors="pt").to(device)
        # Generate the caption
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return JSONResponse(content={"caption": caption})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error generating caption") from e
@app.get("/")
def read_root():
    return {"message": "Welcome to the Image-to-Text Generator API. Use the /generate-caption/ endpoint to generate captions for your images."}