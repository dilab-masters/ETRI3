# ./QAGen/Module/Rationale_generation.py
import json
import re
from typing import Dict, Any
from openai import OpenAI

def generate_rationale_for_qa(
    video_qa: Dict, 
    qa_pair: Dict,
    api_key: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 256
) -> Dict[str, Any]:
    captions_for_prompt = json.dumps(video_qa['OriginalCaptions'], ensure_ascii=False, indent=2)
    qa_pair_for_prompt = json.dumps(qa_pair, ensure_ascii=False, indent=2)
    
    prompt = f"""
##Instruction
You are given a video described by the following captions:

{captions_for_prompt}

Each caption is considered an Event. For the following Problem-AnswerPair, generate a detailed RATIONALE.

Problem-AnswerPair:
{qa_pair_for_prompt}

##Attention:
1. Do NOT change or rephrase the original Answer sentence. Keep it exactly as given.
2. Provide reasoning that clearly explains how the Answer can be inferred from the Events.
3. Explicitly reference the relevant Events by their index (e.g., “Event 0 shows …, Event 2 indicates …”), and connect them logically to support the Answer.
4. Only reference Events that are relevant to reasoning for the Answer; do not include Events that are not necessary for the inference. Do Not include phrases like "as the Answer states"
5. Keep it brief, focused, and do not rephrase the Answer

IMPORTANT: Only provide the explanation text. Do NOT output JSON or extra text.
Example : Event 1 shows the guy removes the pumpkin skin, and Event 3 shows he carves a unique face. This indicates that removing the skin prepares the pumpkin for carving.
"""
    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        reasoning_text = response.choices[0].message.content.strip()
        
        if 'Answer' not in qa_pair:
             qa_pair['Answer'] = {}
             
        qa_pair['Answer']['Rationale'] = reasoning_text

        event_ids = sorted({int(x) for x in re.findall(r'Event (\d+)', reasoning_text)})
        qa_pair["RelatedCaptionsID"] = event_ids

        return qa_pair

    except Exception as e:
        print(f"Error during OpenAI API call or processing: {e}")
        return qa_pair