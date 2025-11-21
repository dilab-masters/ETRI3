#%% /home/dilab/Desktop/Model/VLM.py
import os, json
from tqdm import tqdm
from Module.VLM_inference import load_vlm_model, run_vlm_inference
from config import (BASE_MODEL_ID, ADAPTER_DIR, VIDEO_DIR, QA_JSON_PATH,
                    RESULT_DIR, MERGED_JSON_PATH,
                    NUM_FRAMES, TARGET_SHORT, MAX_NEW_TOKENS, MIN_NEW_TOKENS, FORCE_PREFIX)

os.makedirs(RESULT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(RESULT_DIR, "intermediate_cache.json")

model, processor = load_vlm_model(BASE_MODEL_ID, ADAPTER_DIR)

with open(QA_JSON_PATH, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

print(f"[INFO] Loaded {len(qa_data)} QA pairs from {QA_JSON_PATH}\n")

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    print(f"[INFO] Found existing cache with {len(all_results)} results. Will continue appending...\n")
else:
    all_results = []

for idx, item in enumerate(tqdm(qa_data, desc="Processing QA videos", ncols=100)):
    video_id = str(item["VideoID"])
    question = item["question"]
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        continue

    options_text = ""
    if "options" in item and item["options"]:
        options_text = "Options:\n" + "\n".join([f"{k}. {v}" for k, v in item["options"].items()])

    full_question = f"Question: {question}\n{options_text}"

    try:
        captions, rationale = run_vlm_inference(
            model,
            processor,
            video_path,
            full_question,
            num_frames=24,
            target_short=TARGET_SHORT,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            force_prefix=FORCE_PREFIX,
        )

        result = {
            "VideoID": video_id,
            "question": question,
            "options": item.get("options", {}),
            "answer": item.get("answer", ""),
            "dimension": item.get("dimension", ""),
            "captions": captions,
            "rationale": rationale,
        }

        all_results.append(result)

        if len(all_results) % 10 == 0:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"[CACHE SAVED] {len(all_results)} items saved to {CACHE_PATH}")

    except Exception as e:
        print(f"[ERROR] {video_id} failed: {e}")

with open(MERGED_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\nAll QA videos processed successfully!")
print(f"Merged results saved to: {MERGED_JSON_PATH}")
print(f"Final cache saved to: {CACHE_PATH}")
