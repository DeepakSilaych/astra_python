from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import time
import traceback
import logging
from typing import Optional, Dict, Any
import sys
import os
from virtual_tryon.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

class VirtualTryOnRequest(BaseModel):
    jewelry_image_url: str
    model_image_url: str
    jewelry_type: str
    jewelry_subtype: str
    jewelry_size: Optional[Dict[str, Any]] = None

class VirtualTryOnResponse(BaseModel):
    success: bool
    processed_jewelry_url: str = ""
    processed_jewelry_with_bg_url: str = ""
    landmark_position: Optional[Dict[str, Any]] = None
    temp_file_ids: Optional[list] = []
    processing_time: float = 0.0
    error: str = None

router = APIRouter(prefix="/virtual-try-on", tags=["virtual-try-on"])

@router.post("/process")
async def process_virtual_try_on(request: VirtualTryOnRequest):
    try:
        logger.info(f"[VirtualTryOn] Processing request: {request.jewelry_type}/{request.jewelry_subtype}")
        
        processor = ImageProcessor()
        
        result = await processor.process_virtual_try_on(
            jewelry_image_url=request.jewelry_image_url,
            model_image_url=request.model_image_url,
            jewelry_type=request.jewelry_type,
            jewelry_subtype=request.jewelry_subtype,
            jewelry_size=request.jewelry_size
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Virtual try-on processing failed")
        
        logger.info(f"[VirtualTryOn] Processing completed successfully. Temp files: {len(result.get('temp_file_ids', []))}")
        
        return {
            "success": True,
            "processed_jewelry_url": result["processed_jewelry_url"],
            "processed_jewelry_with_bg_url": result["processed_jewelry_with_bg_url"],
            "model_image_url": result["model_image_url"],
            "landmark_position": result["landmark_position"],
            "jewelry_size": result.get("jewelry_size"),
            "temp_file_ids": result.get("temp_file_ids", []),
            "processing_info": result.get("processing_info", {})
        }
        
    except Exception as e:
        logger.error(f"[VirtualTryOn] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {str(e)}")
