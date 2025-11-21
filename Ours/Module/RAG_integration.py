# # /home/dilab/Desktop/Model/Module/RAG_integration.py
# import chromadb
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import numpy as np
# from tqdm import tqdm
# from typing import Dict, Any


# def exp_normalize(x):
#     """Softmax 정규화 (확률 분포로 변환)"""
#     x = np.array(x)
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()


# def setup_rag_models(db_directory: str, collection_name: str):
#     """ChromaDB, BGE 임베딩, Reranker 모델 초기화"""
#     try:
#         client = chromadb.PersistentClient(path=db_directory)
#         collection = client.get_collection(name=collection_name)
#         query_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

#         # Reranker 초기화
#         reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
#         reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")

#         # CUDA 가능하면 GPU로 이동
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         reranker_model.to(device)
#         reranker_model.eval()

#         tqdm.write(f"✅ RAG 구성 완료: ChromaDB + BGE + Reranker ({device})")
#         return client, collection, query_model, reranker_tokenizer, reranker_model

#     except Exception as e:
#         tqdm.write(f"⚠️ 모델 초기화 실패: {e}")
#         return None, None, None, None, None


# def run_retrieval(
#     client: chromadb.Client,
#     collection: chromadb.Collection,
#     query_model: SentenceTransformer,
#     query_text: str,
#     n_results: int = 5,
#     reranker_tokenizer=None,
#     reranker_model=None
# ) -> Dict[str, Any]:
#     """ChromaDB 검색 후 Reranker 재정렬"""
#     if not client or not collection or not query_model:
#         return None

#     try:
#         # 1️⃣ 1차 검색 (BGE 임베딩)
#         query_embedding = query_model.encode(["query: " + query_text], convert_to_tensor=False).tolist()
#         results = collection.query(
#             query_embeddings=query_embedding,
#             n_results=n_results * 2,  # 후보 더 많이 가져오기
#             include=["metadatas", "documents", "distances"]
#         )

#         if not results or not results.get("documents") or not results["documents"][0]:
#             return None

#         docs = results["documents"][0]
#         metadatas = results["metadatas"][0]

#         # 2️⃣ Reranker 적용 (있을 경우)
#         if reranker_model and reranker_tokenizer:
#             device = next(reranker_model.parameters()).device
#             pairs = [[query_text, doc] for doc in docs]

#             with torch.no_grad():
#                 inputs = reranker_tokenizer(
#                     pairs,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=512
#                 ).to(device)

#                 logits = reranker_model(**inputs, return_dict=True).logits.view(-1).float()
#                 scores = exp_normalize(logits.cpu().numpy())

#             # 3️⃣ 점수 정렬
#             sorted_idx = np.argsort(scores)[::-1]
#             sorted_docs = [docs[i] for i in sorted_idx[:n_results]]
#             sorted_metas = [metadatas[i] for i in sorted_idx[:n_results]]
#             sorted_dist = [float(1 - scores[i]) for i in sorted_idx[:n_results]]

#             return {
#                 "documents": [sorted_docs],
#                 "metadatas": [sorted_metas],
#                 "distances": [sorted_dist],
#             }

#         # Reranker 없이 기본 반환
#         return results

#     except Exception as e:
#         tqdm.write(f"⚠️ RAG 검색 오류: {e}")
#         return None
    
# def extract_wiki_content_url_distance(retrieval_results: dict):
#     wiki_info_list = []
#     if retrieval_results and "documents" in retrieval_results and retrieval_results["documents"]:
#         docs = retrieval_results["documents"][0]
#         urls = retrieval_results["metadatas"][0]
#         distances = retrieval_results.get("distances", [[]])[0]
#         for i in range(len(docs)):
#             wiki_info_list.append({
#                 "text": docs[i],
#                 "url": urls[i].get("url", "N/A"),
#                 "distance": distances[i] if i < len(distances) else None
#             })
#     return wiki_info_list

import chromadb
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any


def setup_rag_models(db_directory: str, collection_name: str):
    try:
        client = chromadb.PersistentClient(path=db_directory)
        collection = client.get_collection(name=collection_name)
        query_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tqdm.write(f"RAG setup completed: ChromaDB + BGE ({device})")
        return client, collection, query_model

    except Exception as e:
        tqdm.write(f"Model initialization failed: {e}")
        return None, None, None


def run_retrieval(
    client: chromadb.Client,
    collection: chromadb.Collection,
    query_model: SentenceTransformer,
    query_text: str,
    n_results: int = 5
) -> Dict[str, Any]:
    if not client or not collection or not query_model:
        return None

    try:
        query_embedding = query_model.encode(["query: " + query_text], convert_to_tensor=False).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )

        if not results or not results.get("documents") or not results["documents"][0]:
            return None

        return results

    except Exception as e:
        tqdm.write(f"RAG retrieval error: {e}")
        return None

    
def extract_wiki_content_url_distance(retrieval_results: dict):
    wiki_info_list = []
    if retrieval_results and "documents" in retrieval_results and retrieval_results["documents"]:
        docs = retrieval_results["documents"][0]
        urls = retrieval_results["metadatas"][0]
        distances = retrieval_results.get("distances", [[]])[0]
        for i in range(len(docs)):
            wiki_info_list.append({
                "text": docs[i],
                "url": urls[i].get("url", "N/A"),
                "distance": distances[i] if i < len(distances) else None
            })
    return wiki_info_list