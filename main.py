from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import traceback
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not installed, using system environment variables")
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from reflection.routes import router as reflection_router
from virtual_tryon.routes import router as tryon_router

app = FastAPI(title="Astra Python Microservices", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Astra Python Microservices",
        "status": "running",
        "services": ["reflection", "virtual_try_on"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": ["reflection", "virtual_try_on"]
    }

app.include_router(reflection_router)
app.include_router(tryon_router)

@app.get("/test")
async def test_endpoint():
    try:
        return {
            "status": "API is working",
            "services": ["reflection", "virtual_try_on"],
            "python_path": sys.path[:3],
            "current_dir": os.getcwd(),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/files/{filename}")
async def serve_file(filename: str):
    try:
        file_path = f"img/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        return FileResponse(
            path=file_path,
            media_type="image/png",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Astra Python Microservices")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path[:3]}")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        logger.info("OpenAI API key configured")
    else:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        logger.info("Create a .env file with OPENAI_API_KEY=your_key_here")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
