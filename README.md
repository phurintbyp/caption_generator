# Image Caption Generator with LLaVA

A web-based image caption generator built with **FastAPI** and a pretrained **LLaVA-OneVision** vision-language model.  
Users can upload an image, generate one or multiple captions, and choose between different caption styles such as standard captioning or object-focused captioning.

## Features

- Upload an image from your computer
- Drag-and-drop image upload
- Image preview before submission
- Generate **1, 3, or 5 captions**
- Choose caption mode:
  - **Standard**: general image caption
  - **Object-focused**: caption emphasizing the main visible objects
- Display uploaded image in the results section
- Copy each caption with one click
- Loading spinner while captions are being generated
- Clean and responsive user interface

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript, Jinja2
- **Model:** `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
- **Libraries:** PyTorch, Transformers, Pillow

## Project Structure

```text
caption_generator/
├── app/
│   ├── main.py
│   ├── model.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
├── .gitignore
├── requirements.txt
└── README.md