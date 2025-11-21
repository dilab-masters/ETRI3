#%%
import json
import torch
from pathlib import Path
from config import VLM_JSON_PATH, OUTPUT_JSON_PATH, DB_DIR, COLLECTION_NAME
from Module.RAG_integration import setup_rag_models, run_retrieval, extract_wiki_content_url_distance
from tqdm import tqdm

def extract_overview_line(captions: str) -> str:
    if not captions:
        return ""
    parts = captions.split("\n")
    for i, line in enumerate(parts):
        if line.strip().startswith("### Event Captions"):
            if i + 1 < len(parts):
                return parts[i + 1].strip()
    return parts[0].strip() if parts else ""


rag_client, rag_collection, rag_model = setup_rag_models(DB_DIR, COLLECTION_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not all([rag_client, rag_collection, rag_model]):
    raise RuntimeError("RAG model/DB initialization failed")

with open(VLM_JSON_PATH, "r", encoding="utf-8") as f:
    vlm_data = json.load(f)

final_data = []

for vid_entry in tqdm(vlm_data, desc="RAG Context Integration"):
    captions_text = vid_entry.get("captions", "")
    question_text = vid_entry.get("question", "")

    overview_text = extract_overview_line(captions_text)
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

print(f"Context integration completed! Results saved: {OUTPUT_JSON_PATH}")
# %%
