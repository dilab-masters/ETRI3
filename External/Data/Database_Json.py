import json
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def chunk_text(text, max_tokens=256, overlap_tokens=20):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

def process_jsonl_files_and_store_chroma(
    input_jsonl_files,
    chroma_db_path="./e5_wiki_chroma_db",
    collection_name="e5_wiki_articles_collection",
    embedding_model_name="BAAI/bge-small-en-v1.5",
    batch_size=64
):
    print(f"Starting JSONL file processing and ChromaDB storage")
    
    client = chromadb.PersistentClient(path=chroma_db_path)
    
    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    print(f"Embedding model loaded. (Device: {model.device})")

    collection = client.get_or_create_collection(name=collection_name)
    print(f"Adding data to ChromaDB collection '{collection_name}'")
    print(f"Current number of documents in collection: {collection.count()}")

    current_batch_texts = []
    current_batch_metadatas = []
    current_batch_ids = []
    
    total_chunks_added = 0

    for input_file_path in input_jsonl_files:
        if not os.path.exists(input_file_path):
            print(f"Warning: File '{input_file_path}' not found. Skipping.")
            continue

        print(f"\nProcessing file: '{input_file_path}'")

        try:
            current_file_total_lines = sum(1 for line in open(input_file_path, 'r', encoding='utf-8'))
        except Exception as e:
            print(f"Error: Cannot calculate line count for '{input_file_path}'. {e}")
            continue

        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line_idx, line in enumerate(tqdm(infile, total=current_file_total_lines, desc=f"Processing {os.path.basename(input_file_path)}")):
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_idx+1} in '{input_file_path}': {e}. Skipping this line.")
                    continue
                
                doc_id = record.get('id', f'unknown_id_{line_idx}')
                doc_title = record.get('title', 'Unknown Title')
                doc_url = record.get('url', '')
                doc_text = record.get('text', '')

                if doc_text:
                    chunks = chunk_text(doc_text)
                    for j, chunk_text_content in enumerate(chunks):
                        chunk_id = f"{doc_id}-{j}"
                        metadata = {
                            "doc_id": doc_id,
                            "title": doc_title,
                            "url": doc_url,
                            "chunk_index": j
                        }
                        
                        current_batch_texts.append(chunk_text_content)
                        current_batch_metadatas.append(metadata)
                        current_batch_ids.append(chunk_id)

                        if len(current_batch_texts) >= batch_size:
                            texts_for_embedding = ["passage: " + text for text in current_batch_texts]
                            embeddings = model.encode(texts_for_embedding, convert_to_tensor=False).tolist()
                            
                            try:
                                collection.add(
                                    documents=current_batch_texts,
                                    metadatas=current_batch_metadatas,
                                    embeddings=embeddings,
                                    ids=current_batch_ids
                                )
                                total_chunks_added += len(current_batch_texts)
                            except Exception as e:
                                print(f"Warning: Error adding batch to ChromaDB: {e}")
                                print(f"Error batch starting from ID: {current_batch_ids[0]}...")

                            current_batch_texts = []
                            current_batch_metadatas = []
                            current_batch_ids = []

    if current_batch_texts:
        texts_for_embedding = ["passage: " + text for text in current_batch_texts]
        embeddings = model.encode(texts_for_embedding, convert_to_tensor=False).tolist()
        try:
            collection.add(
                documents=current_batch_texts,
                metadatas=current_batch_metadatas,
                embeddings=embeddings,
                ids=current_batch_ids
            )
            total_chunks_added += len(current_batch_texts)
        except Exception as e:
            print(f"Warning: Error adding final batch to ChromaDB: {e}")
            print(f"Error batch starting from ID: {current_batch_ids[0]}...")

    print(f"\nCompleted processing all JSONL files. Total {collection.count()} chunks added to ChromaDB.")
    print(f"ChromaDB saved to: '{chroma_db_path}'")


if __name__ == "__main__":
    base_jsonl_dir = "./Data_Json/"
    chroma_db_directory = "./bge_wiki_chroma_db"
    chroma_collection_name = "bge_wiki_rag_collection"

    input_jsonl_file_paths = [
        os.path.join(base_jsonl_dir, "ds_data_1.jsonl"),
        os.path.join(base_jsonl_dir, "ds_data_2.jsonl"),
    ]

    process_jsonl_files_and_store_chroma(
        input_jsonl_file_paths,
        chroma_db_path=chroma_db_directory,
        collection_name=chroma_collection_name
    )