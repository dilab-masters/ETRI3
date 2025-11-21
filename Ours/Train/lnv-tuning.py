import os, json, torch, numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import (
    LlavaNextVideoForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader

DATA_JSON = "./Train/Training_dataset.json"
BASE_MODEL = "llava-hf/LLaVA-NeXT-Video-7B-hf"
SAVE_DIR = "LLaVA_NeXT_Video_7B_lora"

def resize_short(img: Image.Image, target_short: int = 336) -> Image.Image:
    w, h = img.size
    if min(w, h) == target_short:
        return img
    if w <= h:
        nw = target_short
        nh = int(h * (target_short / w))
    else:
        nh = target_short
        nw = int(w * (target_short / h))
    return img.resize((nw, nh), Image.BILINEAR)

def load_video_uniform_frames(video_path: str, num_frames: int = 16, target_short: int = 336):
    try:
        import decord
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            return []
        idxs = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
        batch = vr.get_batch(idxs).asnumpy()
        frames = [Image.fromarray(x) for x in batch]
        return [resize_short(f, target_short) for f in frames]
    except Exception:
        pass
    
    try:
        import av
        container = av.open(video_path)
        all_frames = [f.to_image() for f in container.decode(video=0)]
        container.close()
        if not all_frames:
            return []
        idxs = np.linspace(0, len(all_frames) - 1, num=min(num_frames, len(all_frames)), dtype=int)
        return [resize_short(all_frames[int(i)], target_short) for i in idxs]
    except Exception:
        pass
    
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total == 0:
            cap.release()
            return []
        idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = bgr[:, :, ::-1]
            frames.append(resize_short(Image.fromarray(rgb), target_short))
        cap.release()
        return frames
    except Exception:
        return []

def format_data(sample):
    return [
        {"role": "system", "content": [{"type": "text", "text": sample["texts"]["system"]}]},
        {"role": "user", "content": [
            {"type": "video", "video": sample["video"]},
            {"type": "text", "text": sample["texts"]["user"]},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": sample["texts"]["assistant"]}]},
    ]

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb,
    attn_implementation="eager",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

processor.tokenizer.padding_side = "right"
processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()

for n, p in model.named_parameters():
    ln = n.lower()
    if ("vision" in ln) or ("visual" in ln) or ("multi_modal_projector" in ln) or ("mm_projector" in ln):
        p.requires_grad = False

lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
peft_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=lora_targets, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

with open(DATA_JSON, "r", encoding="utf-8") as f:
    raw = json.load(f)
train_dataset = [format_data(s) for s in raw]
print(f"Total samples: {len(train_dataset)}")

def collate_fn(examples):
    full_texts, prompt_texts, video_lists = [], [], []

    for ex in examples:
        full_texts.append(processor.apply_chat_template(ex, tokenize=False, add_generation_prompt=False))
        only_prompt = [turn for turn in ex if turn["role"] != "assistant"]
        prompt_texts.append(processor.apply_chat_template(only_prompt, tokenize=False, add_generation_prompt=True))

        vpath = None
        for turn in ex:
            if turn["role"] == "user":
                for c in turn["content"]:
                    if c.get("type") == "video":
                        vpath = c["video"]
                        break
        frames = load_video_uniform_frames(vpath, num_frames=12, target_short=336)
        if not frames:
            frames = [Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))]
        video_lists.append(frames)

    batch = processor(text=full_texts, videos=video_lists, return_tensors="pt", padding=True)

    with torch.no_grad():
        tok_prompt = processor(text=prompt_texts, return_tensors="pt", padding=True)
    prompt_lens = tok_prompt["attention_mask"].sum(dim=1).tolist()

    labels = batch["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    mm_token_ids = []
    for attr in ("image_token", "video_token"):
        tok = getattr(processor, attr, None)
        if tok is not None:
            try:
                mm_token_ids.append(processor.tokenizer.convert_tokens_to_ids(tok))
            except Exception:
                pass
    if not mm_token_ids:
        mm_token_ids = [151652, 151653, 151655]
    for tid in set(mm_token_ids):
        labels[labels == tid] = -100

    for i, plen in enumerate(prompt_lens):
        labels[i, :int(plen)] = -100

    batch["labels"] = labels
    return batch

BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
EPOCHS = 2
NUM_WORKERS = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
num_update_steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
num_training_steps = EPOCHS * num_update_steps_per_epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

print(f"Effective batch size = {BATCH_SIZE} x {GRAD_ACCUM}")
print(f"Optimizer steps (total) = {num_training_steps}")

global_step = 0
for ep in range(EPOCHS):
    print(f"\nEpoch {ep+1}/{EPOCHS}")
    pbar = tqdm(train_loader, total=len(train_loader))
    total_loss = 0.0

    for step, batch in enumerate(pbar):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k in ("pixel_values", "pixel_values_videos"):
                    batch[k] = v.to(device, dtype=model.config.torch_dtype)
                else:
                    batch[k] = v.to(device)

        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        total_loss += loss.item() * GRAD_ACCUM
        pbar.set_postfix(loss_avg=total_loss / (step + 1))

    print(f"Epoch {ep+1} average loss: {total_loss / len(train_loader):.4f}")

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
processor.tokenizer.save_pretrained(SAVE_DIR)
print(f"\nTraining completed. Saved to: {SAVE_DIR}")