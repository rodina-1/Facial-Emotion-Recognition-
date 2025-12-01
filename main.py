"""
Facial Expression Recognition API
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import timm
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Facial Expression Recognition API",
    description="Detect emotions from facial images using SWIN Transformer",
    version="1.0.0"
)

# CORS - Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
DEVICE = None
CLASS_NAMES = None
EXPRESSION_MAP = {
    '1': 'Surprise',
    '2': 'Fear',
    '3': 'Disgust',
    '4': 'Happy',
    '5': 'Sad',
    '6': 'Angry',
    '7': 'Neutral'
}

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model():
    """Load the trained SWIN model"""
    global MODEL, DEVICE, CLASS_NAMES
    
    try:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {DEVICE}")
        
        # Create model architecture
        MODEL = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
        MODEL.head = nn.Linear(MODEL.head.in_features, 7)  # 7 emotion classes
        
        # Load trained weights
        checkpoint = torch.load("best_swin_dataset.pth", map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        CLASS_NAMES = checkpoint.get('class_names', ['1', '2', '3', '4', '5', '6', '7'])
        
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.warning("⚠️  Model not loaded. Place 'best_swin_dataset.pth' in the same directory.")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Facial Expression Recognition API",
        "status": "running",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "unknown"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "unknown"
    }


@app.post("/api/v1/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        JSON with prediction results
    """
    
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure 'best_swin_dataset.pth' is available."
        )
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess image
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            output = MODEL(input_tensor)
            
            # Handle different output shapes
            if output.dim() > 2:
                output = output.mean(dim=[-1, -2])  # Global average pooling
            
            # Get probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence_values = probs[0].cpu().numpy()
            
            # Get top prediction
            pred_idx = probs.argmax(1).item()
            pred_class_number = CLASS_NAMES[pred_idx]
            pred_emotion = EXPRESSION_MAP.get(pred_class_number, pred_class_number)
            pred_confidence = float(confidence_values[pred_idx])
            
            # Get all emotions with confidence scores
            all_emotions = []
            for idx, class_num in enumerate(CLASS_NAMES):
                emotion_name = EXPRESSION_MAP.get(class_num, class_num)
                all_emotions.append({
                    "emotion": emotion_name,
                    "confidence": float(confidence_values[idx])
                })
            
            # Sort by confidence
            all_emotions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return JSONResponse({
            "success": True,
            "prediction": {
                "emotion": pred_emotion,
                "confidence": pred_confidence
            },
            "all_predictions": all_emotions,
            "metadata": {
                "model": "SWIN Transformer",
                "device": str(DEVICE)
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict emotions from multiple images
    
    Args:
        files: List of image files
        
    Returns:
        JSON with batch predictions
    """
    
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    results = []
    
    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = MODEL(input_tensor)
                if output.dim() > 2:
                    output = output.mean(dim=[-1, -2])
                
                probs = torch.nn.functional.softmax(output, dim=1)
                pred_idx = probs.argmax(1).item()
                pred_class_number = CLASS_NAMES[pred_idx]
                pred_emotion = EXPRESSION_MAP.get(pred_class_number, pred_class_number)
                pred_confidence = float(probs[0][pred_idx].cpu().numpy())
            
            results.append({
                "file_index": idx,
                "filename": file.filename,
                "emotion": pred_emotion,
                "confidence": pred_confidence,
                "success": True
            })
            
        except Exception as e:
            results.append({
                "file_index": idx,
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return JSONResponse({
        "success": True,
        "total_processed": len(results),
        "results": results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)