#%% ./Ours/LLM.py
import os
import numpy as np
from collections import defaultdict
from Module.LLM_inference import run_llm_eval
from config import OUTPUT_JSON_PATH, RESULT_DIR, QWEN_OUTPUT

os.makedirs(RESULT_DIR, exist_ok=True)

env = "w/o_rationale" # only_llm, ours, w/o_rag, w/o_rationale
model_key = "qwen"
model_output_file = QWEN_OUTPUT
model_name_str = "Qwen/Qwen2.5-7B-Instruct"
num_runs = 5

accuracies = []
all_dim_acc = []

print(f"Starting {num_runs} consecutive {env} evaluations. (Model: {model_key})")
print(f"Input file (Rationale + Context): {OUTPUT_JSON_PATH}")

for i in range(num_runs):
    print("\n" + "="*50)
    print(f"--- Run {i + 1}/{num_runs} started ---")
    print("="*50)
    
    base_output_filename = os.path.basename(model_output_file)
    output_filename = f"hybrid_results_{base_output_filename}_run_{i+1}.json"
    RUN_OUTPUT_FILE_PATH = os.path.join(RESULT_DIR, output_filename)

    print(f"Output file (LLM evaluation): {RUN_OUTPUT_FILE_PATH}")

    results, overall_acc, dim_acc = run_llm_eval(
        qa_json_path=OUTPUT_JSON_PATH,
        output_json_path=RUN_OUTPUT_FILE_PATH,
        env=env,
        model_name=model_name_str
    )

    print(f"\n--- Run {i + 1} completed! ---")
    print(f"Run {i+1} Overall Hybrid Accuracy: {overall_acc*100:.2f}%")
    
    accuracies.append(overall_acc)
    all_dim_acc.append(dim_acc)

avg_accuracy_percent = np.mean(accuracies) * 100
std_dev_percent = np.std(accuracies) * 100

print("\n" + "="*50)
print(f"Completed {num_runs} runs. Final results summary")
print("="*50)
print(f"Accuracy per run: {[f'{acc*100:.2f}%' for acc in accuracies]}")
print(f"Average accuracy: {avg_accuracy_percent:.2f}%")
print(f"Standard deviation (σ): {std_dev_percent:.2f}%")

print("\nAverage dimension-wise accuracy over 5 runs:")
avg_dims = defaultdict(list)
for dim_run in all_dim_acc:
    for dim, val in dim_run.items():
        if dim == "Unknown": 
            continue
        avg_dims[dim].append(val['accuracy'])

for dim, acc_list in avg_dims.items():
    avg_dim_acc = np.mean(acc_list) * 100
    print(f" - {dim}: {avg_dim_acc:.2f}%")