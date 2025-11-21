# /home/dilab/Desktop/Model/Module/VLM_inference.py
import os, torch, numpy as np
from PIL import Image
from transformers import LlavaNextVideoForConditionalGeneration, AutoProcessor
from peft import PeftModel
import re, json


def resize_short(img: Image.Image, target_short: int = 336) -> Image.Image:
    w, h = img.size
    if min(w, h) == target_short:
        return img
    if w <= h:
        nw = target_short; nh = int(h * (target_short / w))
    else:
        nh = target_short; nw = int(w * (target_short / h))
    return img.resize((nw, nh), Image.BILINEAR)

def load_video_uniform_frames(video_path: str, num_frames: int = 16, target_short: int = 336):
    try:
        import decord
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total > 0:
            idxs = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
            batch = vr.get_batch(idxs).asnumpy()  # (N,H,W,3)
            frames = [Image.fromarray(x) for x in batch]
            return [resize_short(f, target_short) for f in frames]
    except Exception:
        pass
    try:
        import av
        container = av.open(video_path)
        all_frames = [f.to_image() for f in container.decode(video=0)]
        container.close()
        if all_frames:
            idxs = np.linspace(0, len(all_frames)-1, num=min(num_frames, len(all_frames)), dtype=int)
            return [resize_short(all_frames[int(i)], target_short) for i in idxs]
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total == 0:
            cap.release(); return []
        idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if not ok: continue
            rgb = bgr[:, :, ::-1]
            frames.append(resize_short(Image.fromarray(rgb), target_short))
        cap.release()
        return frames
    except Exception:
        return []

def load_vlm_model(base_model_id: str, adapter_dir: str):
    dtype = torch.float16
    base = LlavaNextVideoForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    try:
        processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
        proc_src = adapter_dir
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
        proc_src = base_model_id
    print(f"[INFO] VLM processor : {proc_src}")

    processor.tokenizer.padding_side = "right"
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, processor

def run_vlm_inference(
    model: PeftModel, 
    processor: AutoProcessor, 
    video_path: str, 
    question: str,
    num_frames: int = 16,
    target_short: int = 336,
    max_new_tokens: int = 512,
    min_new_tokens: int = 96,
    force_prefix: bool = True
):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    frames = load_video_uniform_frames(video_path, num_frames=num_frames, target_short=target_short)
    if not frames:
        raise RuntimeError("Failed to decode video into frames.")

    # 2. 입력 준비 (프롬프트)
    SYSTEM_PROMPT = "Given a video and a question, first generate descriptive every event captions in detail from the video. Then, using these captions, provide a Only rationale that answers the question."
    USER_TEXT = f"Question: {question}\n"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "video", "video": "<video>"}, {"type": "text", "text": USER_TEXT}]},
    ]

    if force_prefix:
        prompt_core = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ANSWER_PREFIX = "### Event Captions\n0: "
        prompt = prompt_core + ANSWER_PREFIX
    else:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[prompt], videos=[frames], return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    bad_words = []
    if processor.tokenizer.unk_token_id is not None:
        bad_words = [[processor.tokenizer.unk_token_id]]

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature = 0.7,
            repetition_penalty=1.05,
            bad_words_ids=bad_words or None,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

    in_len  = inputs["input_ids"].shape[1]
    trimmed = [out[in_len:] for out in gen]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    assistant_out = (decoded[0] if decoded else "").strip()

    separator = "### Rationale"
    captions = ""
    rationale = ""

    if separator in assistant_out:
        try:
            parts = assistant_out.split(separator, 1)
            captions = parts[0].strip()
            if not captions.startswith("### Event Captions"):
                 captions = "### Event Captions\n" + captions
            rationale = separator + "\n" + parts[1].strip()
        except Exception:
            captions = assistant_out if assistant_out else "(empty)"
            rationale = "(Rationale not found)"
    else:
        if not assistant_out.startswith("### Event Captions") and assistant_out:
             captions = "### Event Captions\n" + assistant_out
        else:
             captions = assistant_out if assistant_out else "(empty)"
        rationale = "(Rationale separator not found)"
    
    return captions, rationale

def output_to_json(video_path: str, question: str, captions: str, rationale: str, save_dir: str = "./results"):
    os.makedirs(save_dir, exist_ok=True)

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    caption_dict = {}
    caption_lines = re.findall(r'(\d+):\s*(.+)', captions)
    for idx, text in caption_lines:
        caption_dict[idx] = text.strip()

    if not caption_dict and captions:
        clean = captions.replace("### Event Captions", "").replace("### Captions", "").strip()
        if clean:
            caption_dict["0"] = clean

    clean_rationale = rationale.replace("### Rationale", "").strip()

    result_json = {
        "VideoID": video_id,
        "Captions": caption_dict,
        "Question": question.strip(),
        "Rationale": clean_rationale
    }

    save_path = os.path.join(save_dir, f"{video_id}_result.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"[INFO] JSON 저장 완료 → {save_path}")
    return result_json
