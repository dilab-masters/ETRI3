# ./QAGen/Module/RAG_integration.py
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
from tqdm import tqdm
from typing import List, Dict, Any

def setup_rag_models(db_directory: str, collection_name: str):
    try:
        client = chromadb.PersistentClient(path=db_directory)
        collection = client.get_collection(name=collection_name)
        query_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        tqdm.write("ChromaDB and BAAI embedding model loaded.")
        return client, collection, query_model
    except Exception as e:
        tqdm.write(f"ChromaDB or embedding model initialization failed: {e}")
        return None, None, None

def run_retrieval(
    client: chromadb.Client, 
    collection: chromadb.Collection, 
    query_model: SentenceTransformer, 
    query_text: str, 
    n_results: int = 1
):
    if not client or not collection or not query_model:
        return None
    try:
        query_embedding = query_model.encode(["query: " + query_text], convert_to_tensor=False).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
        return results
    except Exception as e:
        tqdm.write(f"ChromaDB 검색 오류: {e}")
        return None

def extract_wiki_content_url_distance(retrieval_results: dict):
    wiki_info_list = []
    if retrieval_results and 'documents' in retrieval_results and retrieval_results['documents']:
        docs = retrieval_results['documents'][0]
        urls = retrieval_results['metadatas'][0]
        distances = retrieval_results.get('distances', [[]])[0]
        for i in range(len(docs)):
            doc_info = {
                "text": docs[i],
                "url": urls[i].get('url', 'N/A'),
                "distance": distances[i] if i < len(distances) else None
            }
            wiki_info_list.append(doc_info)
    return wiki_info_list

def generate_reasoning_with_rag(
    openai_client: OpenAI,
    model: str,
    OriginalCaptions_str: str, 
    Question: str, 
    Answer: str, 
    Event_rationale: str,
    Event_ids: list,
    wiki_content: str
):
    prompt = f"""
##Instruction:
Your task is to combine the **-level evidence (captions)** and the **retrieved Wiki text** to generate a single, final rationale. The rationale must clearly explain how the content of multiple frames follows a temporal or logical sequence.

##Event-level Captions:
{OriginalCaptions_str}

##Question:
{Question}

##Answer:
{Answer}

##Event_Rationale:
{Event_rationale}

##Common knowledge:
{wiki_content}

##Event IDs:
{Event_ids}

##Instruction:
Your task is to combine the frame-level evidence (captions) and the common knowledge to generate a single, final rationale. The output must be a valid JSON object with the following two keys: "Answer" and "Rationale". The rationale must clearly explain how the content of multiple frames follows a temporal or logical sequence.
Answer:
1.If the provided answer is correct, the value should be the original answer.
2.If the provided answer is incorrect or incomplete and can be changed by using common sense and the frame evidence, the value should be the corrected or more complete answer.

Rationale:
1.Always explicitly mention the original **Event IDs** in the rationale.
2.Produce only one section.
3.The explanation must be **Simple**, showing exactly which frames support the answer.
4.Instead of mentioning the retrieved Wiki text, integrate the content into the rationale by explaining that it aligns with common knowledge or universal rules.
5.Use the common knowledge only when necessary to concisely provide general context or additional information about the actions in the frames.
6.Output only the two sections: Answer and Rationale. Do not add any meta-comments or other sections.
7.If the provided answer can be changed based on the combined evidence, generate a Answer and explain the change in the rationale.


Example :
Scenario 1: The answer needs to be changed.
{{ 
"Answer": "The person is knitting a scarf.",
"Rationale": "In the sequence of events shown in Event 1 (a person holding yarn) and Event 3 (a person holding knitting needles), the actions follow a logical sequence. The common knowledge that knitting requires both yarn and needles confirms that the person is knitting."
}}

Scenario 2: The answer does not need to be changed.
{{
"Answer": "{Answer}",
"Rationale": "The events in Event 3, showing a person pouring water on a plant, followed by Event 5, showing a healthier plant, demonstrate a cause-and-effect relationship. This aligns with the common knowledge that plants need water to thrive." 
}}
"""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
            response_format={"type": "json_object"} # JSON 모드 활성화
        )
        content = response.choices[0].message.content
        return json.loads(content) # 바로 json.loads
    
    except json.JSONDecodeError as e:
        tqdm.write(f"RAG 생성 JSON 디코딩 오류: {e}\n내용: {content[:200]}...")
        return {"error": str(e), "content": content}
    except Exception as e:
        tqdm.write(f"RAG 생성 오류: {e}")
        return {"error": str(e)}

def run_rag_pipeline(
    rationale_data: List[Dict],
    rag_client: chromadb.Client, 
    rag_collection: chromadb.Collection, 
    rag_model: SentenceTransformer,
    api_key: str,
    gpt_model: str
) -> List[Dict]:

    final_data_list = []
    openai_client = OpenAI(api_key=api_key) 

    progress_bar = tqdm(rationale_data, desc="RAG 적용 중")
    for entry in progress_bar:
        video_id = entry['VideoID']
        
        original_captions_list = entry['OriginalCaptions']
        all_captions_str = "\n".join(original_captions_list)
        
        video_entry = {
            "VideoID": video_id, 
            "OriginalCaptions" : original_captions_list, 
            "Problem-AnswerPairs": []
        }

        for qa_pair in entry.get('Problem-AnswerPairs', []):
            question = qa_pair['Question']
            answer = qa_pair['Answer']['Answer']
            frame_rationale = qa_pair['Answer']['Rationale']
            related_ids = qa_pair['RelatedCaptionsID'] 
            
            frames_str = "\n".join([
                f"Event {i}: {original_captions_list[i]}" 
                for i in related_ids if i < len(original_captions_list)
            ])
            
            retrieval_results = run_retrieval(rag_client, rag_collection, rag_model, all_captions_str + " " + question)
            
            if retrieval_results:
                wiki_info = extract_wiki_content_url_distance(retrieval_results)
                wiki_content = "\n\n".join([item['text'] for item in wiki_info])
                wiki_url = wiki_info[0]['url'] if wiki_info else 'N/A'
            else:
                wiki_content = ""
                wiki_url = "N/A"
                
            generated_data_dict = generate_reasoning_with_rag(
                openai_client,
                gpt_model,
                frames_str, 
                question,
                answer,
                frame_rationale,
                related_ids,
                wiki_content
            )
            
            if "error" in generated_data_dict:
                tqdm.write(f"⚠️ RAG 생성 실패 (VideoID: {video_id}): {generated_data_dict['error']}")
                final_answer = answer
                final_rationale = frame_rationale
            else:
                final_answer = generated_data_dict.get('Answer', answer)
                final_rationale = generated_data_dict.get('Rationale', frame_rationale)
            
            video_entry['Problem-AnswerPairs'].append({
                "Question": question,
                "Dimension": qa_pair['Dimension'], 
                "RelatedCaptionsID": related_ids,
                "RelatedFrame": qa_pair.get('RelatedFrame', []),
                "Answer": {
                    "Answer": final_answer,
                    "Rationale": final_rationale,
                    "SourceURL": wiki_url,
                    # "RetrievedContext": wiki_content
                }
            })
        final_data_list.append(video_entry)

    return final_data_list