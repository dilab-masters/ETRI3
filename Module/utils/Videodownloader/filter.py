import json
import csv
import os

def main():
    failed_videos = set()
    failed_csv_path = "./Module/utils/Videodownloader/failed.csv"

    if os.path.exists(failed_csv_path):
        with open(failed_csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    failed_videos.add(row[0].strip())
        print(f"Loaded {len(failed_videos)} failed videos")

    json_path = "./Module/utils/Videodownloader/merged_data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    initial_count = len(data)
    data_cleaned = {}

    for key, value in data.items():
        video_id = key.replace("v_", "")
        if video_id not in failed_videos:
            data_cleaned[key] = value

    removed_count = initial_count - len(data_cleaned)
    print(f"Removed {removed_count} videos")
    print(f"Remaining: {len(data_cleaned)} videos")

    output_path = "./Data/Caption/Data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_cleaned, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()