#%%
import json
from pathlib import Path

training_json_path = "/home/dilab/Desktop/QAGen/Ours/Train/Training_dataset.json"
qa_json_path = "/home/dilab/Desktop/QAGen/Ours/Data/QA_Ours.json"
output_path = "/home/dilab/Desktop/QAGen/Ours/Train/Training_dataset_filtered.json"

with open(training_json_path, "r", encoding="utf-8") as f:
    training_data = json.load(f)

with open(qa_json_path, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

qa_video_ids = set()
for item in qa_data:
    video_id = item.get("VideoID")
    if video_id:
        clean_id = video_id.replace("v_", "")
        qa_video_ids.add(clean_id)

print(f"QA_Ours.json의 고유 VideoID: {len(qa_video_ids)}")
print(f"샘플 VideoID: {list(qa_video_ids)[:5]}\n")

filtered_training_data = []
removed_count = 0
removed_videos = []

for item in training_data:
    video_path = item.get("video", "")
    
    if not video_path:
        filtered_training_data.append(item)
        continue
    
    video_filename = Path(video_path).stem
    clean_filename = video_filename.replace("v_", "")
    
    is_overlap = clean_filename in qa_video_ids
    
    if is_overlap:
        removed_count += 1
        removed_videos.append(video_filename)
        print(f"제거됨: {video_filename}")
    else:
        filtered_training_data.append(item)

print(f"\n원본 Training_dataset.json: {len(training_data)} 개")
print(f"필터링 후 Training_dataset.json: {len(filtered_training_data)} 개")
print(f"제거된 겹치는 항목: {removed_count} 개")

if removed_videos:
    print(f"\n제거된 비디오:")
    for vid in removed_videos[:10]:
        print(f"  - {vid}")
    if len(removed_videos) > 10:
        print(f"  ... 외 {len(removed_videos) - 10}개")

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_training_data, f, ensure_ascii=False, indent=2)

print(f"\n필터링된 데이터셋 저장: {output_path}")
# %%
