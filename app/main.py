from io import BytesIO
from pathlib import Path
import base64

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.model import generate_captions

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "captions": None,
            "error": None,
            "filename": None,
            "selected_mode": "standard",
            "selected_count": 1,
            "image_data": None,
            "image_mime": None,
        },
    )


@app.post("/caption", response_class=HTMLResponse)
async def caption_image(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("standard"),
    count: int = Form(1),
):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            return templates.TemplateResponse(
                request,
                "index.html",
                {
                    "captions": None,
                    "error": "Please upload a valid image file.",
                    "filename": None,
                    "selected_mode": mode,
                    "selected_count": count,
                    "image_data": None,
                    "image_mime": None,
                },
            )

        if count not in [1, 3, 5]:
            count = 1

        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        max_size = 1024
        image.thumbnail((max_size, max_size))

        captions = generate_captions(image=image, mode=mode, num_captions=count)

        image_data = base64.b64encode(contents).decode("utf-8")

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "captions": captions,
                "error": None,
                "filename": file.filename,
                "selected_mode": mode,
                "selected_count": count,
                "image_data": image_data,
                "image_mime": file.content_type,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "captions": None,
                "error": str(e),
                "filename": None,
                "selected_mode": mode,
                "selected_count": count,
                "image_data": None,
                "image_mime": None,
            },
        )