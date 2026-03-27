from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
CACHE_DIR = r"D:\hf_cache"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model = None
processor = None


def load_model():
    global model, processor

    if model is None or processor is None:
        print("Loading model once...")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR
        )
        processor.tokenizer.padding_side = "left"

    return model, processor


def build_prompt(mode: str) -> str:
    prompts = {
        "standard": "Write one short, natural caption for this image.",
        "objects": "Write one short caption focused on the main visible objects in this image. Clearly mention the important objects and keep it to one sentence.",
    }
    return prompts.get(mode, prompts["standard"])


def generate_captions(image: Image.Image, mode: str = "standard", num_captions: int = 1):
    model, processor = load_model()
    image = image.convert("RGB")

    prompt_text = build_prompt(mode)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(device)
            if device == "cuda" and inputs[k].dtype.is_floating_point:
                inputs[k] = inputs[k].to(dtype)

    generation_kwargs = {
        "max_new_tokens": 60,
    }

    if num_captions > 1:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 0.9,
            "num_return_sequences": num_captions,
        })
    else:
        generation_kwargs.update({
            "do_sample": False,
        })

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

    cleaned = []
    seen = set()

    for text in decoded:
        caption = text.strip()
        if caption and caption not in seen:
            seen.add(caption)
            cleaned.append(caption)

    return cleaned