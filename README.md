# 시공간 정보 결합을 통한 관계 이해 및 저장 관리 기술
---
An automated system for generating and filtering Question-Answer (QA) pairs and rationales from video content. This project provides a comprehensive pipeline for creating high-quality QA datasets with supporting evidence from video materials.

## Feature
- External Knowledge Integration
  - We use external knowledge to answer questions that the video alone cannot explain.
 
    
- Two-Stage Architecture (Rationale-Based)
  - A vision model explains the 'reasoning', and then a language model gives the final answer
 
    
- Caption-Based QA Dataset
  - We turn video events into text captions to help the AI understand the story's flow and context.

```
ETRI3/
├── Module/                     # QA 생성 모듈 
│   └── utils/                 
├── Ours/                       # 커스텀 VLM/LLM
│   ├── Train/
│   │   ├── lnv-tuning.py       # LLaVA-NeXT-Video 학습습 스크립트
│   │   └── Training_dataset.json # 학습용 데이터셋 포맷
│   ├── Data_QA/
│   ├── Module/                 # Tuned VLM/LLM 추론 관련 모듈듈
│   ├── config.py               # Tuned VLM/LLM 전역 설정 파일
│   └── run_pipeline.py         # Tuned VLM/LLM 추론 파이프라인 스크립트트
├── config.py                   # QA 생성 프로젝트 전역 설정 파일
├── Main.py                     # QA 생성 파이프라인인 스크립트트
└── demo.py                     # 데모 시연용 스크립트
```

### 1. Environment
- Create conda virtual env
```bash
git clone https://github.com/dilab-masters/ETRI3.git
cd ETRI3
conda create -n ETRI python=3.13.5
conda activate ETRI
pip install -r requirements.txt
```
```bash
# create .env
OPENAI_API_KEY="Your_OPENAI_API_KEY"
```

- Data Download (Video)

We used the [ActivityNet Captions dataset](https://cs.stanford.edu/people/ranjaykrishna/densevid/) for dense video captioning

```bash
cd Module/utils/Videodownloader/
python youtube_downloader.py --json input.json --out /Your_path/ETRI3/Data/Video
python filter.py
cd Your_path/ETRI3/Data/Video
python Crop.py
```

- Data Download (External Knowledge)

We used the [wikipedia 20220301](https://www.kaggle.com/datasets/nbroad/wiki-20220301-en) for external knowledge

```bash
# From Kagglehub / Need Kagglehub AP
cd Your_Path/ETRI3/External/Data/
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
python enwiki.py                 # wiki-20220301-en
python Database_Json.py          # VectorDB
```

### 2. QA Generation 

A process that utilizes dense video captions to automatically generate high-quality QA pairs requiring temporal and causal reasoning, going beyond simple factual verification.

```python
# config.py
DEFAULT_FILE_NAME = "" # Caption Json Data Name 
BASE_DIR = "./Data"
CHROMA_COLLECTION_NAME = "Your_RAG_COLLECTION"

cd Your_path/ETRI3/
python Main.py
```
```
# /ETRI3/Data/QA
- QA : Initial QA pairs generated from video captions
- QA_Filter : QA pairs after filtering
- QA_Final_RAG : Final QA pairs enhanced with external knowledge (Final dataset)
- QA_Rationale : QA pairs with generated rationales (reasoning explanations)
```

Example 
```json
  {
    "VideoID": "v_XD3yFrJHiv8",
    "OriginalCaptions": {
      "0": "A man and woman are sitting on red couches with sports jerseys hanging on the wall and a blue screen on the wall behind them in the middle of them.",
      "1": "The woman picks up a clipboard next to her and smooths out the papers that are on top of it while smiling, crosses her legs and sits back.",
      "2": "The man is talking this whole time while the woman gets comfortable on her couch and the woman briefly talks.",
      "3": "Various clips begin to play and it briefly shows parts of a city, then most of it is of men outdoors and playing soccer on sand in a middle of city, and occasionally blue words pop up on the screen when certain plays in the game are made.",
      "4": "When the clips end the man and woman on the couch begin talking and she puts her clipboard down."
    },
"Problem-AnswerPairs": [
      {
        "Question": "What might have happened if the woman had not picked up the clipboard?",
        "Question_Type": "Counterfactual",
        "Event_Caption": [
          1,
          2
        ],
        "Answer": "If the woman had not picked up the clipboard, she might not have organized the papers or appeared as prepared during the discussion.",
        "Rationale_With_external_data": {
          "Rationale": "Event 1 shows the woman picking up a clipboard and smoothing out the papers on top of it, which suggests she is organizing them and preparing for the discussion. Event 2 indicates that the man is talking while the woman gets comfortable on her couch and briefly talks, implying that she is participating in the discussion. If the woman had not picked up the clipboard, she might not have organized the papers or appeared as prepared during the discussion, as the act of organizing the papers is directly linked to her preparation.",
          "wiki_url": "https://en.wikipedia.org/wiki/Keeping%20Score%20%28TV%20series%29"
        }
      },
      {
        "Question": "What is likely to happen after the man and woman start talking again?",
        "Question_Type": "Predictive",
        "Event_Caption": [
          2,
          3,
          4
        ],
        "Answer": "After the man and woman start talking again, they are likely to discuss the events or plays shown in the soccer clips.",
        "Rationale_With_external_data": {
          "Rationale": "Event 2 describes the man talking while the woman gets comfortable, indicating a discussion or presentation setting. Event 3 shows clips of men playing soccer with blue words appearing on the screen during certain plays, highlighting significant moments likely tied to the discussion. Event 4 shows the man and woman resuming their conversation after the clips, suggesting they are likely to discuss the soccer clips' content, as it is the most recent and relevant topic. This aligns with common knowledge that such clips are often used to facilitate discussions in sports presentations.",
          "wiki_url": "https://en.wikipedia.org/wiki/Virtual%20replay"
        }
      }
    ]
  }
```
### 2-1. QA Generation Demo
Demo
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/47bf56c0-63e1-4b03-88f5-64ae586e23c5" width="100%" />
      <br />
      <sub>Input Video</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/6061a2bd-5132-405e-9529-f3643ebb225e" width="100%" />
      <br />
      <sub>QA Generation</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/25f18814-19bb-46e7-a3f3-0bc3041de7ce" width="100%" />
      <br />
      <sub>Key Frame Selection</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/c8af2956-3db0-465e-aa8b-de3306a80a14" width="100%" />
      <br />
      <sub>Rationale Generation & Final Results Check</sub>
    </td>
  </tr>
</table>


```bash
cd ETRI3
python demo.py
```


### 3. Ours Model
A two-stage framework that fine-tunes a VLM to generate visual rationales and synthesizes the final answer by integrating these rationales with retrieved external knowledge via an LLM.

### 3-1. Training
We fine-tune the [LLaVA-NeXT-Video](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) model using the transformed dataset.
```bash
# ETRI3/Ours/Train/Training_dataset.json
  {
    "video": "/your_path/Data/Video/VideoID.mp4",
    "texts": {
      "system": "Given a video and a question, first generate descriptive event captions from the video. Then, using these captions, provide a rationale that answers the question.",
      "user": "Question: ...",
      "assistant": "### Event Captions\n0: ... \n\n### Rationale\n..."
    }
  }
```

```bash
cd Ours/Train
python lnv-tuning.py
```

### 3-2. Test
Evaluation on the MCQ Dataset Generated by the QA Generation Pipeline

- Data Download (Video)

```bash
cd ..
cd Data_QA 
python youtube_downloader.py --json QA_Ours.json --out Video_Ours
```

STAGE 1: VLM VIDEO INFERENCE → STAGE 2: RAG CONTEXT INTEGRATION → STAGE 3: LLM EVALUATION
```bash
cd ..
python run_pipeline.py --env ours --num-runs 5
```
```bash
# env = 'Ours'
======================================================================
FINAL RESULTS SUMMARY
======================================================================
Runs: ['55.98%', '55.98%', '57.42%', '56.94%', '56.94%']
Average: 56.65%
Std Dev: 0.57%

Dimension-wise Accuracy:
  Causal: 54.29%
  Counterfactual: 41.54%
  Factual: 67.73%
  Predictive: 67.80%
  Temporal: 50.70%

======================================================================
PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
```

```bash
# env = 'w/o_rationale'
======================================================================
FINAL RESULTS SUMMARY
======================================================================
Runs: ['53.59%', '52.63%', '54.07%', '52.63%', '54.07%']
Average: 53.40%
Std Dev: 0.65%

Dimension-wise Accuracy:
  Causal: 50.00%
  Counterfactual: 32.82%
  Factual: 65.91%
  Predictive: 61.95%
  Temporal: 54.42%

======================================================================
PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
```

```bash
# env = 'w/o_rag'
======================================================================
FINAL RESULTS SUMMARY
======================================================================
Runs: ['56.94%', '57.89%', '57.89%', '57.89%', '57.89%']
Average: 57.70%
Std Dev: 0.38%

Dimension-wise Accuracy:
  Causal: 56.19%
  Counterfactual: 47.18%
  Factual: 60.45%
  Predictive: 72.20%
  Temporal: 52.09%
======================================================================
PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
```



### 3-3 Test (NExT-QA)
Evaluation on the [NExT-QA](https://github.com/doc-doc/NExT-QA) Dataset (Download Dataset)
```python
# Ours/Module/Prompt.py
# Using commented-out prompts

# Ours/Module/LLM_inference.py
prompt = prompt_template.format(
    question=question,
    a=opt_filled["a"],
    b=opt_filled["b"],
    c=opt_filled["c"],
    d=opt_filled["d"],
    e=opt_filled["e"],       # add
    captions=captions,
    rationale=rationale,
    context=context
)

# Ours/Module/LLM_inference.py
VIDEO_DIR     = "./Data/Video_NExT" 
QA_JSON_PATH  = "./Data/QA_NExT.json"
```
```bash
python run_pipeline.py --env ours --num-runs 5
```
