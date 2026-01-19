import logging
import os
from pathlib import Path
import shutil

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, JSONResponse

from deepfake.rpc.api_server.deps import get_config, get_rpc
from deepfake.rpc.rpc import RPC

logger = logging.getLogger(__name__)

# Private API, protected by authentication and webserver_mode dependency
router = APIRouter()

@router.get("/uploads/{filename}", tags=["webserver"])
def get_files(filename: str, config: dict = Depends(get_config)):
    """
    Serve uploaded files (video or image) with correct media type.
    """
    UPLOAD_DIR: Path = config["upload_dir"]
    file_path: Path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Detect file type based on extension
    ext = file_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(path=file_path, media_type=media_type, filename=filename)

@router.post("/upload-video", tags=["background"])
async def upload_video(
    user_id: int = Form(...),
    video: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(""),
    duration: str = Form(...),

    rpc: RPC = Depends(get_rpc),
    config: dict = Depends(get_config),
):
    try:
        UPLOAD_DIR = config["upload_dir"]
        file_path: Path = UPLOAD_DIR / video.filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        new_entry = rpc._rpc_add_deepfake_video(
            title=title,
            user_id=user_id,
            description=description,
            duration=duration,
            file_path=str(file_path.resolve()),
            video_filename=video.filename
        )

        return JSONResponse(content={
            "id": new_entry["id"],
            "message": "Video uploaded successfully",
            "video_filename": new_entry["video_filename"],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.post("/upload-image", tags=["background"])
async def upload_video(
    user_id: int = Form(...),
    image: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(""),

    rpc: RPC = Depends(get_rpc),
    config: dict = Depends(get_config),
):
    try:
        UPLOAD_DIR = config["upload_dir"]
        file_path: Path = UPLOAD_DIR / image.filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        new_entry = rpc._rpc_add_deepfake_image(
            title=title,
            user_id=user_id,
            description=description,
            file_path=str(file_path.resolve()),
            image_filename=image.filename
        )

        return JSONResponse(content={
            "id": new_entry["id"],
            "message": "Image uploaded successfully",
            "image_filename": new_entry["image_filename"],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")