'''
@article{yao2025gens,
    title={Generative Frame Sampler for Long Video Understanding},
    author={Yao, Linli and Wu, Haoning and Ouyang, Kun and Zhang, Yuanxing and Xiong, Caiming and Chen, Bei and Sun, Xu and Li, Junnan},
    journal={arXiv preprint arXiv:2503.09146},
    year={2025}
}
'''
# ./QAGen/Module/utils/Key_Frame.py
import torch
from PIL import Image
import os
import glob
from typing import List
import json

from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from qwen_vl_utils import process_vision_info

def load_and_process_images(frame_paths: List[str], max_size: int = 490) -> List[Image.Image]:
    frames = []
    for path in frame_paths:
        try:
            img = Image.open(path).convert("RGB")
            if img.width > max_size or img.height > max_size:
                ratio = min(max_size / img.width, max_size / img.height)
                img = img.resize((int(img.width * ratio), int(img.height * ratio)))
            frames.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return frames

def setup_model(model_id="yaolily/GenS-qwen2d5-vl-3b"):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print(f"GenS Model '{model_id}' loaded successfully!")
    return model, tokenizer, processor

def chunk_frames(frame_paths, chunk_size):
    for i in range(0, len(frame_paths), chunk_size):
        yield frame_paths[i:i + chunk_size]


def gens_frame_sampler(question: str, frame_paths: List[str], model, tokenizer, processor, fps: float = 1.0, chunk_size: int = 1):

    if not frame_paths:
        return "Error: No valid frames provided."

    results = {}
    frame_idx_offset = 0

    for chunk_id, chunk in enumerate(chunk_frames(frame_paths, chunk_size)):
        print(f"Processing chunk {chunk_id+1}: frames ")

        prompt = """Please identify the video frames most relevant to the given question and provide 
              their timestamps in seconds along with a relevance score. The score should be on a 
              scale from 1 to 5, where higher scores indicate greater relevance. Return the output 
              strictly in the following JSON format: {"timestamp (Only "Number - Number")": score, ...}.
            """

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": [f"file://{p}" for p in chunk],
                        "fps": fps,
                    },
                    {"type": "text", "text": f"{prompt}\nQuestion: {question}"}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        ).to("cuda")

        try:
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            try:
                frame_result_json = json.loads(output_text)
                updated_frame_result = {}
                for key, score in frame_result_json.items():
                    start_frame = int(float(key.split("-")[0])) + frame_idx_offset
                    end_frame = int(float(key.split("-")[1])) + frame_idx_offset
                    updated_frame_result[f"{start_frame}-{end_frame}"] = score
                results.update(updated_frame_result)
            except Exception as e:
                print(f"Warning: Failed to parse chunk {chunk_id} result: {e}")

        except Exception as e:
            print(f"Error during chunk {chunk_id} sampling: {e}")

        frame_idx_offset += len(chunk)

    return results


