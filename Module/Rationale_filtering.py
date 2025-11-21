# ./QAGen/Module/Rationale_filtering.py
"""
@inproceedings{golovneva2022roscoe,
   title={{ROSCOE}: A Suite of Metrics for Scoring Step-by-Step Reasoning},
   author={Golovneva, Olga and Chen, Moya and Poff, Spencer and Corredor, Martin and Zettlemoyer, Luke and Fazel-Zarandi, Maryam and Celikyilmaz, Asli},
   year={2022}
}
"""

from __future__ import annotations
import os, re, json, math, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

@dataclass
class RoscoeThresholds:
    SA_Faith_step: Optional[float] = 0.65
    SS_Info_chain: Optional[float] = 0.65
    SS_Rep_step: Optional[float] = 0.01
    LI_Self_cons: Optional[float] = 0.50
    LC_PPL_inv: Optional[float] = 0.01
    LC_Grammar: Optional[float] = 0.60

    d_SA_Faith: Optional[float] = 0.00
    d_SS_Info: Optional[float] = 0.00

    allow_none: bool = True
    require_all: bool = True

@dataclass
class RoscoeConfig:
    cap_max: int = 0
    drop_failures: bool = False
    save_scores_json: Optional[str] = None
    device: Optional[str] = None

def _ensure_hfhub_shim():
    try:
        import huggingface_hub as _hub
    except Exception:
        return
    if not hasattr(_hub, "list_repo_tree"):
        def _shim_list_repo_tree(*args, **kwargs):
            try:
                files = _hub.list_repo_files(*args, **kwargs)
                return [{"path": f} for f in files]
            except Exception:
                return []
        _hub.list_repo_tree = _shim_list_repo_tree
        import sys
        sys.modules["huggingface_hub"].list_repo_tree = _shim_list_repo_tree
    try:
        import huggingface_hub.utils as _hub_utils
        if not hasattr(_hub_utils, "OfflineModeIsEnabled"):
            class OfflineModeIsEnabled(Exception): pass
            _hub_utils.OfflineModeIsEnabled = OfflineModeIsEnabled
            import sys
            sys.modules["huggingface_hub.utils"].OfflineModeIsEnabled = OfflineModeIsEnabled
        if not hasattr(_hub_utils, "is_offline_mode"):
            def is_offline_mode() -> bool: return False
            _hub_utils.is_offline_mode = is_offline_mode
            import sys
            sys.modules["huggingface_hub.utils"].is_offline_mode = is_offline_mode
    except Exception:
        pass

_ensure_hfhub_shim()

def _import_tf():
    from transformers import (
        AutoModel, AutoTokenizer,
        AutoModelForSequenceClassification,
        GPT2LMHeadModel, GPT2TokenizerFast
    )
    return AutoModel, AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast

_SENT_SPLIT_REGEX = re.compile(r'(?<=[\.\?\!])\s+')

def _sent_split(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = _SENT_SPLIT_REGEX.split(text.strip())
    return [p.strip() for p in parts if p and not p.isspace()]

def _to_text(v):
    if v is None: return ""
    if isinstance(v, str): return v
    if isinstance(v, list): return " ".join([_to_text(x) for x in v])
    if isinstance(v, dict): return " ".join([_to_text(x) for x in v.values()])
    return str(v)

def _safe_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def _flatten_examples_from_rationale(rationale_data: list[dict]) -> list[dict]:
    """
    Flatten samples matching Step 3 Rationale output structure:
    - Question: p["Question"] / p["question"]
    - Rationale priority:
        1) p["Rationale_With_external_data"]["Rationale"]
        2) p["Rationale"] / p["rationale"]
        3) p["Answer"]["Rationale"] (Data_with_Rationale.json structure)
        4) Answer dict fields: Reasoning / rationale / Justification …
    """
    exs: list[dict] = []
    total_pairs, miss_q, miss_rat = 0, 0, 0

    for item in rationale_data:
        vid = item.get("VideoID") or item.get("video_id")
        pairs = item.get("Problem-AnswerPairs") or item.get("problem_answer_pairs") or []
        for idx, p in enumerate(pairs):
            total_pairs += 1
            p = p or {}

            q = p.get("Question") or p.get("question")
            if not isinstance(q, str) or not q.strip():
                miss_q += 1
                continue

            ans_val = p.get("Answer") or p.get("answer") or {}
            ans_obj = ans_val if isinstance(ans_val, dict) else {}
            if isinstance(ans_val, str):
                a = ans_val
            else:
                a = (
                    ans_obj.get("Final Answer")
                    or ans_obj.get("final_answer")
                    or ans_obj.get("Answer")
                    or ans_obj.get("answer")
                )

            rat_obj = p.get("Rationale_With_external_data") or {}
            reasoning = (
                rat_obj.get("Rationale")
                or p.get("Rationale")
                or p.get("rationale")
                or (ans_obj.get("Rationale") if isinstance(ans_obj, dict) else None)
                or (ans_obj.get("Reasoning") if isinstance(ans_obj, dict) else None)
                or (ans_obj.get("rationale") if isinstance(ans_obj, dict) else None)
                or (ans_obj.get("Justification") if isinstance(ans_obj, dict) else None)
                or (ans_obj.get("justification") if isinstance(ans_obj, dict) else None)
            )

            rat_steps = _sent_split(_to_text(reasoning))
            if not rat_steps:
                miss_rat += 1
                continue

            exs.append({
                "video_id": vid,
                "pair_index": idx,
                "dimension": p.get("Dimension") or p.get("dimension"),
                "question": q.strip(),
                "answer": _to_text(a).strip() if a else None,
                "rationale": rat_steps,
                "selected_captions": p.get("SelectedCaptions") or p.get("selected_captions") or []
            })

    if len(exs) == 0:
        raise ValueError(
            f"ROSCOE: No valid samples (check Question/Reasoning) — "
            f"total pairs={total_pairs}, missing Question={miss_q}, missing Rationale={miss_rat}"
        )
    return exs

class RoscoeEmbedder:
    def __init__(self, model_name="facebook/roscoe-512-roberta-base", device=None, max_len=512):
        AutoModel, AutoTokenizer, *_ = _import_tf()
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_len = max_len

    @torch.inference_mode()
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batch = self.tok(
            texts, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt"
        ).to(self.device)
        out = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(emb, p=2, dim=1)

def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.matmul(a, b).item()

def load_models(device: Optional[str] = None):
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast = _import_tf()
    emb = RoscoeEmbedder("facebook/roscoe-512-roberta-base", device=device)

    nli_name = "microsoft/deberta-large-mnli"
    nli_tok = AutoTokenizer.from_pretrained(nli_name)
    nli = AutoModelForSequenceClassification.from_pretrained(nli_name).to(emb.device).eval()

    cola_name = "textattack/roberta-base-CoLA"
    cola_tok = AutoTokenizer.from_pretrained(cola_name)
    cola = AutoModelForSequenceClassification.from_pretrained(cola_name).to(emb.device).eval()

    gpt2_name = "gpt2-large"
    gpt2_tok = GPT2TokenizerFast.from_pretrained(gpt2_name)
    if gpt2_tok.pad_token is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name).to(emb.device).eval()

    return emb, (nli_tok, nli), (cola_tok, cola), (gpt2_tok, gpt2), emb.device

@torch.inference_mode()
def _ppl_chain(text, gpt2_tok, gpt2, device):
    enc = gpt2_tok(text, return_tensors="pt").to(device)
    loss = gpt2(**enc, labels=enc["input_ids"]).loss
    ppl = math.exp(loss.item())
    return 1.0 / ppl

@torch.inference_mode()
def _cola_accept(sent, cola_tok, cola, device):
    inputs = cola_tok(sent, return_tensors="pt").to(device)
    logits = cola(**inputs).logits
    return logits.softmax(-1)[0, 1].item()

@torch.inference_mode()
def _nli_contradiction_prob(a, b, nli_tok, nli, device):
    pair = nli_tok(a, b, return_tensors="pt", truncation=True).to(device)
    probs = nli(**pair).logits.softmax(-1)[0]
    return probs[0].item()

def _roscoe_scores(source, hyp_steps, emb_model, nli_pack, cola_pack, gpt2_pack, device):
    e_source = emb_model.encode(source)[0]
    e_steps = emb_model.encode(hyp_steps)
    e_chain = emb_model.encode(" ".join(hyp_steps))[0]

    sims = [_cos_sim(e_steps[i], e_source) for i in range(e_steps.size(0))]
    faith_step = ((sum(sims) / max(1, len(sims))) + 1) / 2

    info_chain = (_cos_sim(e_chain, e_source) + 1) / 2

    rep_max = -1.0
    for i in range(1, e_steps.size(0)):
        si = e_steps[i]
        for j in range(i):
            sj = e_steps[j]
            rep_max = max(rep_max, _cos_sim(si, sj))
    repetition_step = (1 - rep_max) / 2 if rep_max >= -1 else 1.0

    nli_tok, nli = nli_pack
    max_contr = 0.0
    for i in range(1, len(hyp_steps)):
        for j in range(i):
            max_contr = max(max_contr, _nli_contradiction_prob(hyp_steps[i], hyp_steps[j], nli_tok, nli, device))
    self_consistency = 1.0 - max_contr

    cola_tok, cola = cola_pack
    gpt2_tok, gpt2 = gpt2_pack
    try:
        ppl = _ppl_chain(" ".join(hyp_steps), gpt2_tok, gpt2, device)
    except Exception:
        ppl = None
    grams = [_cola_accept(s, cola_tok, cola, device) for s in hyp_steps]
    grammar = sum(grams) / len(grams) if grams else None

    return {
        "SA.FaithfulnessStep": round(faith_step, 4),
        "SS.InfoChain": round(info_chain, 4),
        "SS.RepetitionStep": round(repetition_step, 4),
        "LI.SelfConsistency": round(self_consistency, 4),
        "LC.PerplexityChainInv": round(ppl, 4) if ppl is not None else None,
        "LC.Grammar": round(grammar, 4) if grammar is not None else None,
    }

def _diff(a: dict, b: dict, key: str):
    va, vb = a.get(key), b.get(key)
    return None if (va is None or vb is None) else round(vb - va, 4)

def _meets_thresholds(scores_q: dict, scores_qc: dict, th: RoscoeThresholds) -> bool:
    checks = []
    def chk(val, thr):
        if thr is None: return True
        if val is None: return th.allow_none
        return val >= thr

    checks.append(chk(scores_qc.get("SA.FaithfulnessStep"), th.SA_Faith_step))
    checks.append(chk(scores_qc.get("SS.InfoChain"), th.SS_Info_chain))
    checks.append(chk(scores_qc.get("SS.RepetitionStep"), th.SS_Rep_step))
    checks.append(chk(scores_qc.get("LI.SelfConsistency"), th.LI_Self_cons))
    checks.append(chk(scores_qc.get("LC.PerplexityChainInv"), th.LC_PPL_inv))
    checks.append(chk(scores_qc.get("LC.Grammar"), th.LC_Grammar))

    d_SA = _diff(scores_q, scores_qc, "SA.FaithfulnessStep")
    d_SS = _diff(scores_q, scores_qc, "SS.InfoChain")
    if th.d_SA_Faith is not None:
        checks.append(True if d_SA is None and th.allow_none else (d_SA is not None and d_SA >= th.d_SA_Faith))
    if th.d_SS_Info is not None:
        checks.append(True if d_SS is None and th.allow_none else (d_SS is not None and d_SS >= th.d_SS_Info))

    return all(checks) if th.require_all else any(checks)

def score_and_filter(
    rationale_data: list[dict],
    cfg: RoscoeConfig = RoscoeConfig(),
    th: RoscoeThresholds = RoscoeThresholds(),
) -> tuple[list[dict], list[dict]]:
    """
    Returns:
      - scored_rows: list of flattened scores/deltas/decisions per pair
      - filtered_data: original structure with ROSCOE metadata injected + (optional) failed pairs removed
    """
    examples = _flatten_examples_from_rationale(rationale_data)
    emb, nli_pack, cola_pack, gpt2_pack, device = load_models(cfg.device)

    out_rows: list[dict] = []
    decisions: dict[tuple[Any, int], dict] = {}

    for ex in examples:
        source_q = f"Question: {ex['question']}"
        caps = ex.get('selected_captions') or []
        if cfg.cap_max and isinstance(caps, list):
            caps = caps[:cfg.cap_max]
        caps_text = " ".join([c for c in caps if isinstance(c, str) and c.strip()])
        source_qc = f"Question: {ex['question']} Context: {caps_text}" if caps_text else source_q

        scores_q = _roscoe_scores(source_q, ex["rationale"], emb, nli_pack, cola_pack, gpt2_pack, device)
        scores_qc = _roscoe_scores(source_qc, ex["rationale"], emb, nli_pack, cola_pack, gpt2_pack, device)
        delta = {
            "SA.FaithfulnessStep": _diff(scores_q, scores_qc, "SA.FaithfulnessStep"),
            "SS.InfoChain": _diff(scores_q, scores_qc, "SS.InfoChain"),
        }
        passed = _meets_thresholds(scores_q, scores_qc, th)
        decision = "ROSCOE_PASS" if passed else "ROSCOE_FAIL"

        row = {
            "video_id": ex["video_id"],
            "pair_index": ex["pair_index"],
            "dimension": ex["dimension"],
            "question": ex["question"],
            "answer": ex["answer"],
            "rationale": ex["rationale"],
            "selected_captions": ex["selected_captions"],
            "scores_q_only": scores_q,
            "scores_q_caps": scores_qc,
            "delta": delta,
            "roscoe_decision": decision,
            "roscoe_thresholds": asdict(th),
        }
        out_rows.append(row)
        decisions[(ex["video_id"], ex["pair_index"])] = row

    new_data: list[dict] = []
    for item in rationale_data:
        vid = item.get("VideoID") or item.get("video_id")
        pairs = item.get("Problem-AnswerPairs") or []
        kept = []
        for i, p in enumerate(pairs):
            row = decisions.get((vid, i))
            if not row:
                continue
            p.setdefault("ROSCOE", {})
            p["ROSCOE"]["scores_q_only"] = row["scores_q_only"]
            p["ROSCOE"]["scores_q_caps"] = row["scores_q_caps"]
            p["ROSCOE"]["delta"] = row["delta"]
            p["ROSCOE"]["decision"] = row["roscoe_decision"]
            p["ROSCOE"]["thresholds"] = row["roscoe_thresholds"]
            if cfg.drop_failures and row["roscoe_decision"] == "ROSCOE_FAIL":
                continue
            kept.append(p)
        item["Problem-AnswerPairs"] = kept
        new_data.append(item)

    if cfg.save_scores_json:
        _safe_write_json(out_rows, Path(cfg.save_scores_json))

    return out_rows, new_data