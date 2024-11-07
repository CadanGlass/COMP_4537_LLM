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
    Depends,
    Header,
    Form
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import jwt
from jwt.exceptions import PyJWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta
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
    "https://4537llm.online",                        # Your HTTPS domain
    "https://www.4537llm.online",                    # WWW subdomain
    "https://comp4537-term-project.netlify.app",     # Hosted frontend URL
    "https://spontaneous-vacherin-eda866.netlify.app",  # New frontend URL
    "http://localhost:3000",                         # Local development
    "http://localhost:5173"                          # Vite local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
    ],
    expose_headers=[""],
)



# JWT Configuration


class TokenData(BaseModel):
    username: str
    role: str


# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Sample in-memory database for demonstration purposes
fake_db = {"testuser3": {"username": "testuser3",
                         "password": pwd_context.hash("password123")}}

# Token expiration time
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Helper functions


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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
    print("Received token:", token)  # Debugging: print the received token
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


@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    if username in fake_db:
        raise HTTPException(status_code=400, detail="User already exists")
    hashed_password = pwd_context.hash(password)
    fake_db[username] = {"username": username, "password": hashed_password}
    return {"message": "User registered successfully"}


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = fake_db.get(username)
    if not user or not pwd_context.verify(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"username": username, "role": "user"}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/admin")
async def admin_route(token_data: TokenData = Depends(verify_token)):
    if token_data.role != "admin":
        raise HTTPException(
            status_code=403, detail="Access forbidden: Admins only")
    return {"message": "Welcome, Admin!"}


@app.get("/protected")
async def protected_route(token_data: TokenData = Depends(verify_token)):
    return {"message": f"Hello, {token_data.username}! You have access to this protected route."}


@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...), token_data: TokenData = Depends(verify_token)):
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


@app.post("/logout")
async def logout():
    return {"message": "User logged out successfully"}

# Update the port to 8001 to match your frontend
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
