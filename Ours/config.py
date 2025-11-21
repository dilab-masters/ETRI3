# ./Ours/config.py
"""
cd ..
python run_pipeline.py --env ours --num_runs 5
"""
BASE_MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
ADAPTER_DIR   = "./LLaVA_NeXT_Video_lora"

VIDEO_DIR     = "./Data/Video_Ours" 
QA_JSON_PATH  = "./Data/QA_Ours.json"

RESULT_DIR        = "./JSON"
MERGED_JSON_PATH  = f"{RESULT_DIR}/VLM_CR.json"

NUM_FRAMES     = 16
TARGET_SHORT   = 336
MAX_NEW_TOKENS = 1024
MIN_NEW_TOKENS = 128
FORCE_PREFIX   = True

VLM_JSON_PATH      = f"{RESULT_DIR}/VLM_CR.json"
OUTPUT_JSON_PATH   = f"{RESULT_DIR}/VLM_RAG_Context.json"

DB_DIR             = "../External/Data/bge_wiki_chroma_db"
COLLECTION_NAME    = "bge_wiki_rag_collection"

QWEN_OUTPUT = f"{RESULT_DIR}/Qwen.json"