import json
import os

def merge_and_filter_data(json_files, output_file, min_captions=5, max_per_file=300):
    merged_data = {}
    
    for json_file in json_files:
        if not os.path.exists(json_file):
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            count = 0
            for video_id, video_data in data.items():
                if count >= max_per_file:  # 파일당 300개 제한
                    break
                    
                if isinstance(video_data, dict) and "sentences" in video_data:
                    num_captions = len(video_data.get("sentences", []))
                    
                    if num_captions >= min_captions:
                        merged_data[video_id] = video_data
                        count += 1
        
        except json.JSONDecodeError:
            continue
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved: {len(merged_data)} videos -> {output_file}")

if __name__ == "__main__":
    json_files = [
        "./train_modified.json",
        "./val_1.json",
        "./val_2.json"
    ]
    
    output_file = "../merged_data.json"
    min_captions = 5
    
    merge_and_filter_data(json_files, output_file, min_captions, max_per_file=300)