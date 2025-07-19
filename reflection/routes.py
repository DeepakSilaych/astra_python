from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
import base64
import tempfile
import requests
from celery.result import AsyncResult
from tasks import reflect_task
from celery_worker import celery_app
import io
import os
import reflection.purecv as purecv

router = APIRouter()

@router.post("/reflect")
async def reflect(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(None),
    base64_str: str = Form(None),
    reflection_strength: float = Query(0.4, ge=0.0, le=1.0),
    blur_radius: float = Query(1.0, ge=0.0),
    reflection_y_offset: int = Query(0, ge=0),
    fade_power: float = Query(1.5, ge=0.0),
    sync: bool = Query(False, description="If true, perform reflection synchronously"),
):
    """
    Enqueue an image reflection task and return the Celery task ID.
    """
    try:
        contents = None
        suffix = ".png"
        # Determine source of the image
        if file is not None:
            contents = await file.read()
            suffix = Path(file.filename).suffix or ".png"
        else:
            content_type = request.headers.get("content-type", "")
            # JSON payload case
            if "application/json" in content_type:
                data = await request.json()
                url_body = data.get("url")
                b64_body = data.get("base64")
                if url_body:
                    try:
                        response = requests.get(url_body)
                        response.raise_for_status()
                        contents = response.content
                        suffix = Path(url_body).suffix or ".png"
                    except requests.RequestException as e:
                        raise HTTPException(
                            status_code=400, detail=f"Error fetching URL: {e}"
                        )
                elif b64_body:
                    b64_data = b64_body
                    if b64_data.startswith("data:"):
                        header, b64_data = b64_data.split(",", 1)
                        mime = header.split(";")[0].split(":")[1]
                        ext_map = {
                            "image/jpeg": ".jpg",
                            "image/png": ".png",
                            "image/webp": ".webp",
                            "image/gif": ".gif",
                        }
                        suffix = ext_map.get(mime, ".png")
                    try:
                        contents = base64.b64decode(b64_data)
                    except Exception as e:
                        raise HTTPException(
                            status_code=400, detail=f"Invalid base64 data: {e}"
                        )
                else:
                    raise HTTPException(
                        status_code=400, detail="No file, url, or base64 provided"
                    )
            else:
                # Form-data case without file upload
                if url:
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        contents = response.content
                        suffix = Path(url).suffix or ".png"
                    except requests.RequestException as e:
                        raise HTTPException(
                            status_code=400, detail=f"Error fetching URL: {e}"
                        )
                elif base64_str:
                    b64_data = base64_str
                    if b64_data.startswith("data:"):
                        header, b64_data = b64_data.split(",", 1)
                        mime = header.split(";")[0].split(":")[1]
                        ext_map = {
                            "image/jpeg": ".jpg",
                            "image/png": ".png",
                            "image/webp": ".webp",
                            "image/gif": ".gif",
                        }
                        suffix = ext_map.get(mime, ".png")
                    try:
                        contents = base64.b64decode(b64_data)
                    except Exception as e:
                        raise HTTPException(
                            status_code=400, detail=f"Invalid base64 data: {e}"
                        )
                else:
                    raise HTTPException(
                        status_code=400, detail="No file, url, or base64 provided"
                    )

        # At this point, we have 'contents' and 'suffix'
        # If sync, process immediately without Celery
        if sync:
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = os.path.join(tmpdir, f"input{suffix}")
                output_path = os.path.join(tmpdir, f"output{suffix}")
                with open(input_path, "wb") as f:
                    f.write(contents)
                purecv.generate_reflection_on_original_bg_full_height(
                    image_path=input_path,
                    output_path=output_path,
                    reflection_strength=reflection_strength,
                    blur_radius=blur_radius,
                    reflection_y_offset=reflection_y_offset,
                    fade_power=fade_power,
                )
                with open(output_path, "rb") as f:
                    output_bytes = f.read()
            return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")

        # Otherwise, enqueue Celery task
        task = reflect_task.delay(
            contents,
            suffix,
            reflection_strength,
            blur_radius,
            reflection_y_offset,
            fade_power,
        )
        return {"task_id": task.id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reflect/{task_id}")
async def get_reflection(task_id: str):
    """
    Retrieve the status or result of a reflection task.
    """
    task = AsyncResult(task_id, app=celery_app)
    if task.state in ("PENDING", "STARTED"):
        return JSONResponse({"status": task.state})
    if task.state == "FAILURE":
        return JSONResponse(
            {"status": "FAILURE", "detail": str(task.result)}, status_code=500
        )
    result_b64 = task.result
    output_bytes = base64.b64decode(result_b64)
    return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")