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


def generate_caption(image: Image.Image) -> str:
    model, processor = load_model()
    image = image.convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Write one short, natural caption for this image."},
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

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return caption