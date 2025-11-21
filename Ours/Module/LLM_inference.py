import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from collections import defaultdict
from Module.Prompt import (
    ONLY_LLM_WITH_CAPTION_PROMPT,
    RAG_WITH_RATIONALE_PROMPT,
    RATIONALE_ONLY_PROMPT,
    RAG_WITH_CAPTION_PROMPT
)

def select_prompt(env: str):
    if env == "only_llm":
        return ONLY_LLM_WITH_CAPTION_PROMPT
    elif env == "ours":
        return RAG_WITH_RATIONALE_PROMPT
    elif env == "w/o_rag":
        return RATIONALE_ONLY_PROMPT
    elif env == "w/o_rationale":
        return RAG_WITH_CAPTION_PROMPT
    else:
        raise ValueError(f"Unknown environment: {env}")

def run_llm_eval(
    qa_json_path: str,
    output_json_path: str,
    env: str = "only_llm",
    model_name: str = "lmsys/vicuna-7b-v1.5",
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    print("Model loaded!")

    with open(qa_json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    prompt_template = select_prompt(env)
    results = []

    for item in tqdm(qa_data, desc="LLM Video QA", ncols=100):
        question = item["question"]
        options = item.get("options", {})
        correct_answer = item.get("answer", "").lower()
        dimension = item.get("dimension", "Unknown")

        opt_filled = {k: options.get(k, "") for k in ["a","b","c","d","e"]}
        
        captions = item.get("captions", "")
        rationale = item.get("rationale", "")
        context = item.get("Context", "")

        prompt = prompt_template.format(
            question=question,
            a=opt_filled["a"],
            b=opt_filled["b"],
            c=opt_filled["c"],
            d=opt_filled["d"],
            captions=captions,
            rationale=rationale,
            context=context
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=16,
                top_p=0.95,
                temperature=0.7
            )
        answer = tokenizer.decode(
            gen_tokens[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        if options:
            answer_letter = answer[0] if answer and answer[0] in options else answer
        else:
            answer_letter = answer

        results.append({
            "VideoID": item.get("VideoID", ""),
            "question": question,
            "options": options,
            "ground_truth": correct_answer,
            "model_answer": answer_letter,
            "dimension": dimension
        })

    total = len(results)
    correct = sum(1 for r in results if r["model_answer"] == r["ground_truth"])
    overall_acc = correct / total if total > 0 else 0.0

    dim_acc = defaultdict(lambda: {"correct":0, "total":0})
    for r in results:
        dim = r["dimension"]
        dim_acc[dim]["total"] += 1
        if r["model_answer"] == r["ground_truth"]:
            dim_acc[dim]["correct"] += 1
    for dim in dim_acc:
        c, t = dim_acc[dim]["correct"], dim_acc[dim]["total"]
        dim_acc[dim]["accuracy"] = c / t if t > 0 else 0.0

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_json_path}")
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print("Dimension-wise Accuracy:")
    for dim, val in dim_acc.items():
        print(f" - {dim}: {val['accuracy']*100:.2f}% ({val['correct']}/{val['total']})")

    return results, overall_acc, dim_acc