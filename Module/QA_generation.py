# ./QAGen/Module/QA_generation.py
import json
from typing import List, Dict, Any
from openai import OpenAI

def generate_reasoning_problems_as_is(
    captions: List[str], 
    video_id: str, 
    timestamps: List[List[float]],
    api_key: str,  
    model: str = "gpt-4o"
) -> Any:
    try:
        client = OpenAI(api_key=api_key)
        
        captions_str = "\n".join([f"- {caption}" for caption in captions])
        
        prompt = f"""
## Instruction
Please design diverse multi-image reasoning problem-answer pairs based on the description of the video frame sequence given below.
{captions_str}

## Attention:
1. For Factual and Temporal questions:
   - Factual questions: A question that can be answered using only explicitly stated information within a single Event. (Ex.What is the man wearing?)
   - Temporal questions: Ask about the simple sequence of events using "Before" or "After". The question should be a simple inquiry about the order of actions, without requiring complex inference.
   - For these two types, provide a succinct final answer in one definitive sentence. The question should be very simple and straightforward.

2. For Counterfactual Causal, and Predictive questions:
   - Counterfactual Questions : A counterfactual question prompts reasoning about a hypothetical scenario that did not actually occur. Do NOT list multiple possible outcomes connected by 'or'
   - Predictive Questions : A predictive question requires inferring a future outcome based on the given information and context. Do NOT list multiple possible outcomes connected by 'or'
   - Causal questions: Require a clear explanation of the cause-and-effect relationship, where an action in one caption leads to an action in another. Specifically, include questions that require inferring the unstated 'Why' behind an action or predicting the 'What' of a result.
   - Craft questions that require complex inference, spanning at least two captions, to deduce hidden relationships, intentions, or predict future steps not explicitly stated.
   
3. Not use words like 'caption', 'video' in Question.
4. Ensure that similar objects across events are treated as the same when reasoning across captions.
5. Do NOT generate questions that are highly subjective, including keywords such as: emotional, spiritual, contribution, importance, and implication.

Your output MUST be a valid JSON array that strictly follows the schema below:

[
  {{
    "VideoID": "{video_id}",
    "OriginalCaptions": {json.dumps(captions, ensure_ascii=False)},
    "timestamps": {timestamps},
    "Problem-AnswerPairs": [
      {{
        "Dimension": "Temporal" or "Causal" or "Counterfactual" or "Predictive" or "Factual"
        "Question": "complex multi-image reasoning question",
        "Answer": {{
          "Answer": "one short and succinct sentence answer",
        }}
      }},
      ...
    ]
  }}
]
"""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2048,
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)
    
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: Response is not valid JSON. Error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None