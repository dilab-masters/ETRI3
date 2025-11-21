# ./QAGen/Module/Select_Rationale.py
import os
import json
from typing import List, Dict, Any, Set
from tqdm import tqdm
from .utils.Key_Frame import gens_frame_sampler, setup_model


def save_results(data: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tqdm.write(f"Saved {len(data)} results to {path}")


def map_and_filter_captions(
    video_data: Dict, 
    scores: Any, 
    dimension: str = "", 
    fps: float = 1.0
) -> List[Dict]:
    captions = video_data["OriginalCaptions"]
    timestamps = video_data["timestamps"]
    min_score = 4 if dimension in ["Counterfactual", "Predictive"] else 5

    if isinstance(scores, str):
        try:
            scores = json.loads(scores)
        except Exception:
            return []
            
    if isinstance(scores, list):
        try:
            scores = dict(scores)
        except Exception:
            return []
            
    if not isinstance(scores, dict):
        return []

    selected_caps = []
    for cap, (ts_start, ts_end) in zip(captions, timestamps):
        cap_start_frame = int(ts_start * fps)
        cap_end_frame = int(ts_end * fps)
        total_frames = cap_end_frame - cap_start_frame + 1
        valid_frames = []
        
        for frame_range, score in scores.items():
            if score < min_score:
                continue
                
            try:
                start, end = map(int, frame_range.split("-"))
            except ValueError:
                continue
                
            overlap_start = max(start, cap_start_frame)
            overlap_end = min(end, cap_end_frame)
            
            if overlap_start <= overlap_end:
                valid_frames.extend(range(overlap_start, overlap_end + 1))
        
        score_ratio = len(valid_frames) / total_frames if total_frames > 0 else 0
        selected_caps.append({"caption": cap, "score_ratio": round(score_ratio, 3), "five_frames": valid_frames})

    if sum(c['score_ratio'] for c in selected_caps) == 0:
        return [{"caption": cap, "score_ratio": 1.0, "five_frames": []} for cap in captions]
    
    return [c for c in selected_caps if c["score_ratio"] > 0]


def generate_frame_scores_per_video(
    videos_to_process: List[Dict], 
    existing_results: List[Dict], 
    frame_root: str, 
    model: Any, 
    tokenizer: Any, 
    processor: Any, 
    output_path: str,
    fps: float = 1.0, 
    chunk_size: int = 8, 
    save_interval: int = 100
) -> List[Dict]:
    all_results = existing_results
    
    progress_bar = tqdm(
        enumerate(videos_to_process), 
        total=len(videos_to_process), 
    )

    for i, video_data in progress_bar:
        try:
            video_id = video_data["VideoID"]
            
            frame_dir = os.path.join(frame_root, video_id)
            if not os.path.isdir(frame_dir):
                tqdm.write(f"⚠️ Frame directory not found, skipping: {frame_dir}")
                continue

            frame_paths = sorted([
                os.path.join(frame_dir, f) 
                for f in os.listdir(frame_dir)
                if f.lower().endswith((".jpg", ".png"))
            ])

            video_result = {
                "VideoID": video_id,
                "OriginalCaptions": video_data["OriginalCaptions"],
                "Problem-AnswerPairs": []
            }
            
            for qa_pair in video_data.get("Problem-AnswerPairs", []):
                question = qa_pair["Question"]
                dimension = qa_pair.get("Dimension", "")
                answer_text = qa_pair["Answer"]["Answer"] if isinstance(qa_pair["Answer"], dict) else str(qa_pair["Answer"])
                prompt_text = f"{question}\n{answer_text}"

                frame_scores = gens_frame_sampler(
                    question=prompt_text, 
                    frame_paths=frame_paths, 
                    model=model,
                    tokenizer=tokenizer, 
                    processor=processor, 
                    fps=fps, 
                    chunk_size=chunk_size
                )

                selected_caps = map_and_filter_captions(
                    video_data, frame_scores, dimension=dimension, fps=fps
                )
                
                related_captions = [c["caption"] for c in selected_caps]
                related_frames = sorted(list(set(c for cap in selected_caps for c in cap["five_frames"])))
                
                video_result["Problem-AnswerPairs"].append({
                    "Dimension": dimension, 
                    "Question": question, 
                    "Answer": qa_pair["Answer"],
                    "RelatedCaptions": related_captions, 
                    "RelatedFrame": related_frames
                })

            all_results.append(video_result)
            newly_processed_count = i + 1
            
            if newly_processed_count % save_interval == 0:
                save_results(all_results, output_path)

        except KeyError:
            tqdm.write(f"⚠️ Missing 'VideoID' key, skipping item. Data: {video_data}")
            continue
            
    return all_results