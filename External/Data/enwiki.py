"""
export KAGGLE_USERNAME=<your_username>
export KAGGLE_KEY=<your_api_key>
"""

import kagglehub
from datasets import load_dataset
from pathlib import Path
import json
import os

os.environ["KAGGLEHUB_CACHE"] = "Your_Path/ETRI3/External/Data/Data"

def main():
    path = kagglehub.dataset_download("nbroad/wiki-20220301-en")
    print("Path to dataset files:", path)

    files = list(map(str, Path("./Data/datasets/nbroad/wiki-20220301-en/versions/1").glob("**/*.parquet")))
    ds = load_dataset("parquet", data_files=files, split="train")

    ds_split = ds.train_test_split(test_size=0.5, seed=42)
    ds_1 = ds_split['train']
    ds_2 = ds_split['test']

    output_dir = "./External/Data/Data_Json"
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in [("ds_data_1", ds_1), ("ds_data_2", ds_2)]:
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in split_data:
                json_line = json.dumps(row, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"Dataset successfully converted to '{output_file}'.")

if __name__ == "__main__":
    main()