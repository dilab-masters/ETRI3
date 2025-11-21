import os
import sys
import json
import numpy as np
import argparse
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from config import (
    BASE_MODEL_ID, ADAPTER_DIR, VIDEO_DIR, QA_JSON_PATH,
    RESULT_DIR, MERGED_JSON_PATH, VLM_JSON_PATH, OUTPUT_JSON_PATH,
    NUM_FRAMES, TARGET_SHORT, MAX_NEW_TOKENS, MIN_NEW_TOKENS, FORCE_PREFIX,
    DB_DIR, COLLECTION_NAME, QWEN_OUTPUT
)
from Module.VLM_inference import load_vlm_model, run_vlm_inference
from Module.RAG_integration import setup_rag_models, run_retrieval, extract_wiki_content_url_distance
from Module.LLM_inference import run_llm_eval

def ensure_directories():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("[INFO] Result directory created/verified")

def run_vlm_pipeline():
    print("\n" + "="*70)
    print("STAGE 1: VLM VIDEO INFERENCE")
    print("="*70)
    
    cache_path = os.path.join(RESULT_DIR, "intermediate_cache.json")
    
    model, processor = load_vlm_model(BASE_MODEL_ID, ADAPTER_DIR)
    print("[INFO] VLM model loaded successfully")
    
    with open(QA_JSON_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    print(f"[INFO] Loaded {len(qa_data)} QA pairs from {QA_JSON_PATH}")
    
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"[INFO] Found cache with {len(all_results)} results. Continuing...")
    else:
        all_results = []
    
    for idx, item in enumerate(tqdm(qa_data, desc="VLM Processing", ncols=100)):
        video_id = str(item["VideoID"])
        question = item["question"]
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            print(f"[SKIP] Video not found: {video_path}")
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
                num_frames=NUM_FRAMES,
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
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"[CACHE] {len(all_results)} items saved")
        
        except Exception as e:
            print(f"[ERROR] {video_id} failed: {e}")
            continue
    
    with open(MERGED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] VLM stage complete: {len(all_results)} results")
    print(f"[SAVED] {MERGED_JSON_PATH}")
    
    return all_results

def run_rag_pipeline():
    print("\n" + "="*70)
    print("STAGE 2: RAG CONTEXT INTEGRATION")
    print("="*70)
    
    rag_client, rag_collection, rag_model = setup_rag_models(DB_DIR, COLLECTION_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not all([rag_client, rag_collection, rag_model]):
        raise RuntimeError("RAG model/DB initialization failed")
    
    print(f"[INFO] RAG models initialized on {device}")
    
    if not os.path.exists(VLM_JSON_PATH):
        raise FileNotFoundError(f"VLM output not found: {VLM_JSON_PATH}")
    
    with open(VLM_JSON_PATH, "r", encoding="utf-8") as f:
        vlm_data = json.load(f)
    
    print(f"[INFO] Loaded {len(vlm_data)} VLM results")
    
    final_data = []
    
    for vid_entry in tqdm(vlm_data, desc="RAG Integration", ncols=100):
        captions_text = vid_entry.get("captions", "")
        question_text = vid_entry.get("question", "")
        
        parts = captions_text.split("\n")
        overview_text = ""
        for i, line in enumerate(parts):
            if line.strip().startswith("### Event Captions"):
                overview_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
                break
        if not overview_text:
            overview_text = parts[0].strip() if parts else ""
        
        query_text = f"{overview_text} {question_text}".strip()
        
        retrieval_results = run_retrieval(
            client=rag_client,
            collection=rag_collection,
            query_model=rag_model,
            query_text=query_text,
            n_results=1
        )
        
        if retrieval_results:
            wiki_info = extract_wiki_content_url_distance(retrieval_results)
            context_text = "\n\n".join([item['text'] for item in wiki_info])
            source_url = wiki_info[0]['url'] if wiki_info else "N/A"
        else:
            context_text = ""
            source_url = "N/A"
        
        new_entry = vid_entry.copy()
        new_entry["Context"] = context_text
        new_entry["SourceURL"] = source_url
        final_data.append(new_entry)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    Path(OUTPUT_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SUCCESS] RAG stage complete")
    print(f"[SAVED] {OUTPUT_JSON_PATH}")
    
    return final_data

def run_llm_pipeline(env, num_runs):
    print("\n" + "="*70)
    print("STAGE 3: LLM EVALUATION")
    print("="*70)
    
    model_key = "qwen"
    model_name_str = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"[INFO] Environment: {env}")
    print(f"[INFO] Model: {model_key}")
    print(f"[INFO] Number of runs: {num_runs}")
    print(f"[INFO] Input: {OUTPUT_JSON_PATH}")
    
    accuracies = []
    all_dim_acc = []
    
    for i in range(num_runs):
        print(f"\n--- Run {i + 1}/{num_runs} ---")
        
        output_filename = f"qwen_results_run_{i+1}.json"
        run_output_path = os.path.join(RESULT_DIR, output_filename)
        
        results, overall_acc, dim_acc = run_llm_eval(
            qa_json_path=OUTPUT_JSON_PATH,
            output_json_path=run_output_path,
            env=env,
            model_name=model_name_str
        )
        
        print(f"Run {i+1} Accuracy: {overall_acc*100:.2f}%")
        
        accuracies.append(overall_acc)
        all_dim_acc.append(dim_acc)
    
    avg_accuracy_percent = np.mean(accuracies) * 100
    std_dev_percent = np.std(accuracies) * 100
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"Runs: {[f'{acc*100:.2f}%' for acc in accuracies]}")
    print(f"Average: {avg_accuracy_percent:.2f}%")
    print(f"Std Dev: {std_dev_percent:.2f}%")
    
    print("\nDimension-wise Accuracy:")
    avg_dims = defaultdict(list)
    for dim_run in all_dim_acc:
        for dim, val in dim_run.items():
            if dim != "Unknown":
                avg_dims[dim].append(val['accuracy'])
    
    for dim, acc_list in sorted(avg_dims.items()):
        avg_dim_acc = np.mean(acc_list) * 100
        print(f"  {dim}: {avg_dim_acc:.2f}%")

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline execution for QA generation from video"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ours",
        help="Environment name (default: ours)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of evaluation runs (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FULL PIPELINE EXECUTION")
    print("="*70)
    
    ensure_directories()
    
    try:
        vlm_results = run_vlm_pipeline()
        rag_results = run_rag_pipeline()
        run_llm_pipeline(env=args.env, num_runs=args.num_runs)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()