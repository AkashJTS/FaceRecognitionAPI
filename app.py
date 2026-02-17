# import os
# import cv2
# import numpy as np
# import base64
# from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body, Form
# from fastapi.responses import JSONResponse
# from insightface.app import FaceAnalysis
# import uvicorn
# import asyncio
# from dotenv import load_dotenv

# import logging

# # ------------------------------
# # Logging Configuration
# # ------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# logger = logging.getLogger(__name__)

# # ------------------------------
# # Load .env variables
# # ------------------------------
# load_dotenv()
# AUTH_TOKEN = os.getenv("AUTH_TOKEN", "secret-token-123")  # default fallback

# app = FastAPI(title="Face Compare API")

# # ------------------------------
# # Constants & Temp Folder
# # ------------------------------
# TEMP_FOLDER = "temp"
# os.makedirs(TEMP_FOLDER, exist_ok=True)

# MAX_FILE_SIZE = 8 * 1024 * 1024  # 8 MB
# ALLOWED_FORMATS = ["image/jpeg", "image/png"]

# # ------------------------------
# # Load InsightFace Model
# # ------------------------------
# logger.info("Loading InsightFace model...")
# face_app = FaceAnalysis(name="buffalo_s")
# face_app.prepare(ctx_id=0, det_size=(320, 320))  # CPU, faster detection
# logger.info("InsightFace model loaded successfully")

# # ------------------------------
# # Helper: Base64 to bytes
# # ------------------------------
# def base64_to_bytes(data: str):
#     if data.startswith("data:image/jpeg;base64,"):
#         data = data.replace("data:image/jpeg;base64,", "")
#     elif data.startswith("data:image/png;base64,"):
#         data = data.replace("data:image/png;base64,", "")
#     return base64.b64decode(data)

# # ------------------------------
# # Extract Face Embedding
# # ------------------------------
# def get_embedding(image_bytes):
#     np_img = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
#     if img is None:
#         return None

#     faces = face_app.get(img)
#     if len(faces) == 0:
#         return None

#     # Take largest face if multiple detected
#     face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
#     embedding = np.array(face.embedding, dtype=np.float32)
#     embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity
#     return embedding

# # ------------------------------
# # Compare embeddings using cosine similarity
# # ------------------------------
# def compare_faces(emb1, emb2):
#     similarity = float(np.dot(emb1, emb2))  # cosine similarity (0-1)
#     score = round(similarity * 100, 2)     # convert to percentage
#     return score

# # ------------------------------
# # File validation helper
# # ------------------------------
# def validate_file(file_bytes: bytes, content_type: str):
#     if content_type not in ALLOWED_FORMATS:
#         raise HTTPException(status_code=400, detail="Only JPEG or PNG images allowed")
#     if len(file_bytes) > MAX_FILE_SIZE:
#         raise HTTPException(status_code=400, detail="File size exceeds 8 MB")

# # ------------------------------
# # Token verification helper
# # ------------------------------
# def verify_token(request: Request):
#     auth_header = request.headers.get("Authorization")
#     if not auth_header or auth_header != f"Bearer {AUTH_TOKEN}":
#         raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing token")

# # ------------------------------
# # Upload reference image
# # ------------------------------
# @app.post("/uploadImage")
# async def upload_reference(reference: UploadFile = File(...), request: Request = None):
#     verify_token(request)  # 🔥 verify token

#     try:
#         # timeout handling
#         contents = await asyncio.wait_for(reference.read(), timeout=60)
#     except asyncio.TimeoutError:
#         raise HTTPException(status_code=408, detail="Error: Timeout. Try uploading again")

#     validate_file(contents, reference.content_type)

#     file_path = os.path.join(TEMP_FOLDER, "reference.jpg")
#     with open(file_path, "wb") as f:
#         f.write(contents)

#     return {"status": True, "message": "Reference uploaded successfully"}

# # ------------------------------
# # Compare two images with reference
# # Supports both form-data file uploads and base64 JSON fields
# # ------------------------------
# @app.post("/employeeFaceCompare")
# async def compare(
#     request: Request,
#     file1: UploadFile = File(None),
#     file2: UploadFile = File(None),
#     threshold: float = Body(70, embed=True),
#     file1_base64: str = Body(None),
#     file2_base64: str = Body(None),
# ):
#     try:
#         logger.info("---- Face Compare API Called ----")

#         verify_token(request)
#         logger.info("Token verified")

#         reference_path = os.path.join(TEMP_FOLDER, "reference.jpg")

#         if not os.path.exists(reference_path):
#             logger.warning("Reference image not found")
#             raise HTTPException(status_code=400, detail="Upload reference first")

#         logger.info("Reference image found")

#         # Read reference embedding
#         with open(reference_path, "rb") as f:
#             reference_bytes = f.read()

#         emb_ref = get_embedding(reference_bytes)
#         logger.info("Reference embedding generated")

#         # ------------------------------
#         # FILE 1
#         # ------------------------------
#         if file1_base64:
#             try:
#                 bytes1 = base64_to_bytes(file1_base64)
#                 validate_file(bytes1, "image/jpeg")
#                 logger.info("File1 received via base64")
#             except Exception:
#                 logger.exception("Invalid base64 data for file1")
#                 raise HTTPException(status_code=400, detail="Invalid base64 data for file1")

#         elif file1 is not None:
#             try:
#                 bytes1 = await asyncio.wait_for(file1.read(), timeout=60)
#                 validate_file(bytes1, file1.content_type)
#                 logger.info("File1 received via upload")
#             except asyncio.TimeoutError:
#                 logger.error("Timeout while reading file1")
#                 raise HTTPException(status_code=408, detail="Error: Timeout for file1. Try again")
#         else:
#             logger.warning("file1 missing")
#             raise HTTPException(status_code=400, detail="file1 or file1_base64 is required")

#         emb1 = get_embedding(bytes1)
#         logger.info("File1 embedding generated")

#         # ------------------------------
#         # FILE 2
#         # ------------------------------
#         if file2_base64:
#             try:
#                 bytes2 = base64_to_bytes(file2_base64)
#                 validate_file(bytes2, "image/jpeg")
#                 logger.info("File2 received via base64")
#             except Exception:
#                 logger.exception("Invalid base64 data for file2")
#                 raise HTTPException(status_code=400, detail="Invalid base64 data for file2")

#         elif file2 is not None:
#             try:
#                 bytes2 = await asyncio.wait_for(file2.read(), timeout=60)
#                 validate_file(bytes2, file2.content_type)
#                 logger.info("File2 received via upload")
#             except asyncio.TimeoutError:
#                 logger.error("Timeout while reading file2")
#                 raise HTTPException(status_code=408, detail="Error: Timeout for file2. Try again")
#         else:
#             logger.warning("file2 missing")
#             raise HTTPException(status_code=400, detail="file2 or file2_base64 is required")

#         emb2 = get_embedding(bytes2)
#         logger.info("File2 embedding generated")

#         # ------------------------------
#         # FACE VALIDATION
#         # ------------------------------
#         if emb_ref is None or emb1 is None or emb2 is None:
#             logger.warning("Face detection failed in one or more images")

#             return JSONResponse(
#                 status_code=400,
#                 content={
#                     "status": False,
#                     "message": "Face recognition not successful",
#                     "data": {"face_match": False, "match_score": 0.0}
#                 }
#             )

#         # ------------------------------
#         # COMPARE
#         # ------------------------------
#         score1 = compare_faces(emb_ref, emb1)
#         score2 = compare_faces(emb_ref, emb2)

#         final_score = min(score1, score2)
#         final_match = final_score >= threshold

#         logger.info(f"Final Score: {final_score}, Threshold: {threshold}, Match: {final_match}")

#         return {
#             "status": final_match,
#             "message": "Face recognition successful" if final_match else "Face recognition not successful",
#             "data": {
#                 "face_match": final_match,
#                 "match_score": final_score
#             }
#         }

#     except Exception as e:
#         logger.exception("Unhandled error in employeeFaceCompare")

#         return JSONResponse(
#             status_code=500,
#             content={
#                 "status": False,
#                 "message": "Internal server error",
#                 "data": {"face_match": False, "match_score": 0.0}
#             }
#         )


# # ------------------------------
# # Run FastAPI Server
# # ------------------------------
# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)



import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body, Form
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import uvicorn
import asyncio
from dotenv import load_dotenv

import logging

# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# ------------------------------
# Load .env variables
# ------------------------------
load_dotenv()
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "secret-token-123")

app = FastAPI(title="Face Compare API")

# ------------------------------
# Constants & Temp Folder
# ------------------------------
TEMP_FOLDER = "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_FILE_SIZE = 8 * 1024 * 1024  # 8 MB
ALLOWED_FORMATS = ["image/jpeg", "image/png"]

# ------------------------------
# Load InsightFace Model
# ------------------------------
logger.info("Loading InsightFace model...")
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=0, det_size=(320, 320))
logger.info("InsightFace model loaded successfully")

# ------------------------------
# Helper: Base64 to bytes
# ------------------------------
def base64_to_bytes(data: str):
    if data.startswith("data:image/jpeg;base64,"):
        data = data.replace("data:image/jpeg;base64,", "")
    elif data.startswith("data:image/png;base64,"):
        data = data.replace("data:image/png;base64,", "")
    return base64.b64decode(data)

# ------------------------------
# Extract Face Embedding
# ------------------------------
def get_embedding(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return None

    faces = face_app.get(img)
    if len(faces) == 0:
        return None

    face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
    embedding = np.array(face.embedding, dtype=np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

# ------------------------------
# Compare embeddings using cosine similarity
# ------------------------------
def compare_faces(emb1, emb2):
    similarity = float(np.dot(emb1, emb2))
    score = round(similarity * 100, 2)
    return score

# ------------------------------
# File validation helper
# ------------------------------
def validate_file(file_bytes: bytes, content_type: str):
    if content_type not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images allowed")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 8 MB")

# ------------------------------
# Token verification helper
# ------------------------------
def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing token")

# ------------------------------
# Upload reference image (UNCHANGED)
# ------------------------------

# ------------------------------
# Compare two images directly (file1 vs file2)
# ------------------------------
@app.post("/employeeFaceCompare")
async def compare(
    request: Request,
    file1: UploadFile = File(None),
    file2: UploadFile = File(None),
    threshold: float = Body(70, embed=True),
    file1_base64: str = Body(None),
    file2_base64: str = Body(None),
):
    try:
        logger.info("---- Face Compare API Called ----")

        verify_token(request)
        logger.info("Token verified")

        # ------------------------------
        # FILE 1
        # ------------------------------
        if file1_base64:
            try:
                bytes1 = base64_to_bytes(file1_base64)
                validate_file(bytes1, "image/jpeg")
                logger.info("File1 received via base64")
            except Exception:
                logger.exception("Invalid base64 data for file1")
                raise HTTPException(status_code=400, detail="Invalid base64 data for file1")

        elif file1 is not None:
            try:
                bytes1 = await asyncio.wait_for(file1.read(), timeout=60)
                validate_file(bytes1, file1.content_type)
                logger.info("File1 received via upload")
            except asyncio.TimeoutError:
                logger.error("Timeout while reading file1")
                raise HTTPException(status_code=408, detail="Error: Timeout for file1. Try again")
        else:
            raise HTTPException(status_code=400, detail="file1 or file1_base64 is required")

        emb1 = get_embedding(bytes1)
        logger.info("File1 embedding generated")

        # ------------------------------
        # FILE 2
        # ------------------------------
        if file2_base64:
            try:
                bytes2 = base64_to_bytes(file2_base64)
                validate_file(bytes2, "image/jpeg")
                logger.info("File2 received via base64")
            except Exception:
                logger.exception("Invalid base64 data for file2")
                raise HTTPException(status_code=400, detail="Invalid base64 data for file2")

        elif file2 is not None:
            try:
                bytes2 = await asyncio.wait_for(file2.read(), timeout=60)
                validate_file(bytes2, file2.content_type)
                logger.info("File2 received via upload")
            except asyncio.TimeoutError:
                logger.error("Timeout while reading file2")
                raise HTTPException(status_code=408, detail="Error: Timeout for file2. Try again")
        else:
            raise HTTPException(status_code=400, detail="file2 or file2_base64 is required")

        emb2 = get_embedding(bytes2)
        logger.info("File2 embedding generated")

        if emb1 is None or emb2 is None:
            logger.warning("Face detection failed in one or more images")

            return JSONResponse(
                status_code=400,
                content={
                    "status": False,
                    "message": "Face recognition not successful",
                    "data": {"face_match": False, "match_score": 0.0}
                }
            )

        # ------------------------------
        # COMPARE file1 vs file2
        # ------------------------------
        final_score = compare_faces(emb1, emb2)
        final_match = final_score >= threshold

        logger.info(f"Final Score: {final_score}, Threshold: {threshold}, Match: {final_match}")

        return {
            "status": final_match,
            "message": "Face recognition successful" if final_match else "Face recognition not successful",
            "data": {
                "face_match": final_match,
                "match_score": final_score
            }
        }

    except Exception:
        logger.exception("Unhandled error in employeeFaceCompare")

        return JSONResponse(
            status_code=500,
            content={
                "status": False,
                "message": "Internal server error",
                "data": {"face_match": False, "match_score": 0.0}
            }
        )

# ------------------------------
# Run FastAPI Server
# ------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
