import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY")

GPT_MODEL = "gpt-4o"
DEFAULT_FILE_NAME = "Data"

BASE_DIR = "./Data"

CAPTION_DIR = os.path.join(BASE_DIR, "Caption")
QA_DIR = os.path.join(BASE_DIR, "QA", "QA")

QA_FILTER_DIR = os.path.join(BASE_DIR, "QA", "QA_Filter")

QA_RATIONALE_DIR = os.path.join(BASE_DIR, "QA", "QA_Rationale")

QA_RATIONALE_FILTER_DIR = os.path.join(BASE_DIR, "QA", "QA_Rationale_filter")

QA_RAG_DIR = os.path.join(BASE_DIR, "QA", "QA_Final_RAG")

FRAME_ROOT = os.path.join(BASE_DIR, "Video", "Data", "Frame")

os.makedirs(CAPTION_DIR, exist_ok=True)
os.makedirs(QA_DIR, exist_ok=True)
os.makedirs(QA_FILTER_DIR, exist_ok=True)
os.makedirs(QA_RATIONALE_DIR, exist_ok=True)
os.makedirs(QA_RATIONALE_FILTER_DIR, exist_ok=True)
os.makedirs(QA_RAG_DIR, exist_ok=True)

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

QA_FILTER_SUFFIX = "_QA_filtered.json"

ROSCOE_SCORE_SUFFIX = "_roscoe_scores.json"
RATIONALE_VALIDATED_SUFFIX = "_with_Rationale_validated.json"

CHROMA_DB_DIRECTORY = os.path.join(BASE_DIR, "External", "Data", "bge_wiki_chroma_db")
CHROMA_COLLECTION_NAME = "bge_wiki_rag_collection"


def get_file_paths(file_name: str) -> dict:
    return {
        "CAPTION_INPUT": os.path.join(CAPTION_DIR, f"{file_name}.json"),

        "QA_OUTPUT": os.path.join(QA_DIR, f"{file_name}_QA.json"),

        "QA_FILTER_OUTPUT": os.path.join(QA_FILTER_DIR, f"{file_name}{QA_FILTER_SUFFIX}"),

        "SCORE_OUTPUT": os.path.join(QA_DIR, f"{file_name}_FrameScores.json"),

        "RATIONALE_OUTPUT": os.path.join(QA_RATIONALE_DIR, f"{file_name}_with_Rationale.json"),

        "ROSCOE_SCORES": os.path.join(QA_RATIONALE_FILTER_DIR, f"{file_name}{ROSCOE_SCORE_SUFFIX}"),
        "RATIONALE_VALIDATED": os.path.join(QA_RATIONALE_FILTER_DIR, f"{file_name}{RATIONALE_VALIDATED_SUFFIX}"),

        "RAG_INPUT": os.path.join(QA_RATIONALE_FILTER_DIR, f"{file_name}{RATIONALE_VALIDATED_SUFFIX}"),

        "RAG_OUTPUT": os.path.join(QA_RAG_DIR, f"{file_name}_Final_RAG.json"),

        "FRAME_ROOT": FRAME_ROOT,
    }

FPS = 1.0
CHUNK_SIZE = 8
SAVE_INTERVAL = 100

RATIONALE_TEMPERATURE = 0.0
RATIONALE_MAX_TOKENS = 256