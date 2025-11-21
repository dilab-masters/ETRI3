# ./QAGen/Module/QA_filtering.py
from __future__ import annotations
import json, re, os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

VIDEOESPRESSO_QA_PROMPT = """
Use the provided criteria flag to assess each QA pair for quality.
You must rely only on the provided Captions (observable content) and the narrative flow.
Do not assume any external knowledge beyond these Captions.

Captions:
{captions}

Question:
{question}

Answer:
{answer}

For each QA pair flagged as low-quality, provide a brief explanation indicating which criterion was violated
(e.g., "Subjective Question", "Lack of Continuity", "Overly Open-Ended Question", "Off-Caption", "Ambiguous").
Ensure high-quality QA pairs maintain alignment with the video's observable content, narrative flow, and context.

Respond ONLY in JSON with exactly these keys:
{{
  "verdict": "ACCEPT" | "REWRITE" | "REJECT",
  "violations": [
    "Subjective Question" |
    "Lack of Continuity" |
    "Overly Open-Ended Question" |
    "Off-Caption" |
    "Ambiguous"
  ],
  "explanation": "one-line reason"
}}
""".strip()

@dataclass
class FilterConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    cap_max_captions: int = 0
    cap_max_chars: int = 6000
    cap_use_referenced_only: bool = True
    max_new_tokens: int = 192
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    drop_rejects: bool = False
    device_map: str | dict = "auto"

def _to_text(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, str): return v
    if isinstance(v, list):  return " ".join([_to_text(x) for x in v])
    if isinstance(v, dict):  return " ".join([_to_text(x) for x in v.values()])
    return str(v)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _find_brace_span(s: str) -> Optional[Tuple[int,int]]:
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return (start, i + 1)
    return None

def _robust_json_extract(raw: str) -> Optional[Dict[str, Any]]:
    s = _strip_code_fences(raw)
    try:
        return json.loads(s)
    except Exception:
        pass
    span = _find_brace_span(s)
    if span:
        chunk = s[span[0]:span[1]]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

def _truncate_text(text: str, cap_max_chars: int) -> str:
    if cap_max_chars and len(text) > cap_max_chars:
        return text[:cap_max_chars]
    return text

def _caps_dict_to_list(caps: Any, cap_max_captions: int) -> List[str]:
    if isinstance(caps, list):
        lst = [c for c in caps if isinstance(c, str) and c.strip()]
        return lst[:cap_max_captions] if cap_max_captions else lst
    if isinstance(caps, dict):
        items = []
        for k, v in caps.items():
            try:
                idx = int(k)
            except Exception:
                idx = 10**9
            if isinstance(v, str) and v.strip():
                items.append((idx, v))
        items.sort(key=lambda x: x[0])
        lst = [v for _, v in items]
        return lst[:cap_max_captions] if cap_max_captions else lst
    txt = _to_text(caps).strip()
    return [txt] if txt else []

def _select_event_captions(all_caps: List[str], indices: List[int]) -> List[str]:
    sel = []
    for i in indices:
        if 0 <= i < len(all_caps):
            sel.append(all_caps[i])
    return sel if sel else all_caps

def _hard_drop_rejects(data: list[dict]) -> list[dict]:
    new = []
    for item in data:
        pairs = item.get("Problem-AnswerPairs") or []
        kept = []
        for p in pairs:
            if p.get("Decision") == "REJECT_BY_FILTER":
                continue
            kept.append(p)
        item["Problem-AnswerPairs"] = kept
        new.append(item)
    return new

_tokenizer = None
_model = None
_EOS_IDS: List[int] | None = None

def _ensure_model(cfg: FilterConfig):
    global _tokenizer, _model, _EOS_IDS
    if _tokenizer is not None and _model is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map=cfg.device_map, trust_remote_code=True)
    _EOS_IDS = []
    for tok_str in ["<|im_end|>", _tokenizer.eos_token]:
        if tok_str:
            ids = _tokenizer.encode(tok_str, add_special_tokens=False)
            if ids:
                _EOS_IDS.append(ids[0])
    _EOS_IDS = list(dict.fromkeys(_EOS_IDS)) or None

def _call_llm(cfg: FilterConfig, question: str, answer: str, captions_text: str) -> Dict[str, Any]:
    _ensure_model(cfg)
    sys_msg = (
        "You are a strict JSON generator. "
        "Return ONLY a single JSON object with the required keys, no extra text, no markdown."
    )
    user_msg = VIDEOESPRESSO_QA_PROMPT.format(
        captions=captions_text, question=question, answer=answer
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    out = _model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        do_sample=cfg.do_sample,
        eos_token_id=_EOS_IDS[0] if _EOS_IDS else None,
        pad_token_id=_tokenizer.eos_token_id or _tokenizer.pad_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].size(1):]
    raw_text = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    data = _robust_json_extract(raw_text)
    if data is None:
        return {"verdict": "REWRITE", "violations": ["ParseError"], "explanation": "Non-JSON or invalid JSON"}
    v = (data.get("verdict") or "").upper()
    if v not in ("ACCEPT", "REWRITE", "REJECT"):
        v = "REWRITE"
    violations = data.get("violations") or []
    if not isinstance(violations, list):
        violations = []
    return {"verdict": v, "violations": violations, "explanation": data.get("explanation") or ""}

def _build_caption_context(all_caps: List[str], indices: List[int], cfg: FilterConfig) -> str:
    if cfg.cap_use_referenced_only and indices:
        caps = _select_event_captions(all_caps, [int(i) for i in indices if isinstance(i, int) or str(i).isdigit()])
    else:
        caps = all_caps
    return _truncate_text(" ".join(caps), cfg.cap_max_chars)

def _extract_pairs_from_item(item: Dict[str, Any], cfg: FilterConfig) -> List[Dict[str, Any]]:
    vid = item.get("VideoID") or item.get("video_id")
    caps_raw = item.get("OriginalCaptions") or item.get("original_captions") or []
    caps_list = _caps_dict_to_list(caps_raw, cfg.cap_max_captions)

    pairs = item.get("Problem-AnswerPairs") or item.get("problem_answer_pairs") or []
    out = []
    for k, p in enumerate(pairs):
        q = (p or {}).get("Question") or (p or {}).get("question")
        a = (p or {}).get("Answer") or (p or {}).get("answer") or ""
        qtype = (p or {}).get("Question_Type") or (p or {}).get("question_type")
        ev_idx = (p or {}).get("Event_Caption") or (p or {}).get("event_caption_indices") or []
        rat_obj = (p or {}).get("Rationale_With_external_data") or {}
        rationale = rat_obj.get("Rationale", "") or (p or {}).get("Rationale") or (p or {}).get("rationale", "")

        if not (isinstance(q, str) and q.strip()):
            continue
        if not isinstance(a, str):
            a = _to_text(a)

        out.append({
            "video_id": vid,
            "pair_index": k,
            "question_type": qtype,
            "question": q.strip(),
            "answer": a.strip(),
            "original_captions": caps_list,
            "event_caption_indices": ev_idx,
            "rationale": rationale,
        })
    return out

def run_filter_on_memory_data(
    rationale_data: List[Dict[str, Any]],
    cfg: FilterConfig
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for item in rationale_data:
        pairs.extend(_extract_pairs_from_item(item, cfg))

    decisions_by_vid: Dict[Any, List[Tuple[int, Dict[str, Any]]]] = {}
    for ex in pairs:
        captions_text = _build_caption_context(ex["original_captions"], ex["event_caption_indices"], cfg)
        qa_filter = _call_llm(cfg, ex["question"], ex["answer"], captions_text)
        decision = (
            "ACCEPT"
            if qa_filter["verdict"] == "ACCEPT"
            else ("REJECT_BY_FILTER" if qa_filter["verdict"] == "REJECT" else "REWRITE_BY_FILTER")
        )
        rec = {"filter": qa_filter, "decision": decision}
        decisions_by_vid.setdefault(ex["video_id"], []).append((ex["pair_index"], rec))

    new_data: List[Dict[str, Any]] = []
    for item in rationale_data:
        vid = item.get("VideoID") or item.get("video_id")
        pairs = item.get("Problem-AnswerPairs") or []
        injected = []
        dec_map = {idx: meta for idx, meta in decisions_by_vid.get(vid, [])}
        for i, p in enumerate(pairs):
            meta = dec_map.get(i)
            if not meta:
                continue
            p["Filter"] = meta["filter"]
            p["Decision"] = meta["decision"]
            if cfg.drop_rejects and meta["decision"] == "REJECT_BY_FILTER":
                continue
            injected.append(p)
        item["Problem-AnswerPairs"] = injected
        new_data.append(item)
    return new_data

def save_json(data: Any, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)