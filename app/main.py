from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.model import generate_caption

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
            "caption": None,
            "error": None,
            "filename": None,
        },
    )


@app.post("/caption", response_class=HTMLResponse)
async def caption_image(request: Request, file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            return templates.TemplateResponse(
                request,
                "index.html",
                {
                    "caption": None,
                    "error": "Please upload a valid image file.",
                    "filename": None,
                },
            )

        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        caption = generate_caption(image)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "caption": caption,
                "error": None,
                "filename": file.filename,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "caption": None,
                "error": str(e),
                "filename": None,
            },
        )