import time, json, os, shutil   
import gradio as gr
from tqdm import tqdm

from config_demo import (
    get_file_paths,
    OPENAI_API_KEY,
    GPT_MODEL,
    RATIONALE_TEMPERATURE,
    RATIONALE_MAX_TOKENS,
    FPS,
    CHUNK_SIZE,
    SAVE_INTERVAL,
    DEFAULT_FILE_NAME,
    CHROMA_DB_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    BASE_DIR
)
from Module.QA_generation import generate_reasoning_problems_as_is
from Module.Rationale_generation import generate_rationale_for_qa
from Module.Rationale_select import generate_frame_scores_per_video
from Module.utils.Key_Frame import setup_model
from Module.Crop import ensure_frames_for_video
from Module.QA_filtering import (
    FilterConfig,
    run_filter_on_memory_data,
    save_json,
)
from Module.Rationale_filtering import (
    RoscoeThresholds,
    RoscoeConfig,
    score_and_filter,
)
from Module.RAG_integration import setup_rag_models, run_rag_pipeline

VIDEO_SAVE_DIR = os.path.join(BASE_DIR, "Video", "Upload")
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

CSS = r"""
body,
.gradio-container {
  background: #f3f3f3;
  font-family: 'Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic',
               system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
.gradio-container * {
  font-family: 'Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic',
               system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}

#page_wrap { padding: 24px 16px 48px;}

#upload_section, #preview_box, #result_wrap, #loading_wrap {
  width: clamp(420px, 94vw, 1100px);
  margin: 0 auto;
}

#dropzone {
  height: 600px;
  border: 3px dashed #111;
  border-radius: 36px;
  background: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  box-shadow: 0 8px 24px rgba(0,0,0,.06);
}
#dropzone .inner { user-select: none; }
#dropzone svg {
  width: 128px; height: 128px;
  stroke: #111; fill: none; stroke-width: 8px;
}
#dropzone .label {
  margin-top: 10px;
  font-size: 18px;
  color: #111;
}
#upload_section { position: relative; }
#file_overlay {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}
#upload_section.dragover #dropzone {
  border-color: #0a66c2;
  box-shadow: 0 10px 28px rgba(10,102,194,.18);
}

#preview_box{
  background:#fff;
  border:3px dashed #111;
  border-radius:36px;
  padding:28px;
  box-shadow:0 8px 24px rgba(0,0,0,.06);
}
#video_comp video, #video_comp .wrap {
  border-radius:16px;
  overflow:hidden;
}
#gen_btn {
  width:220px;
  margin:18px auto 0;
  display:block;
  font-weight:700;
}

#loading_wrap {
  background:#fff;
  border:3px dashed #111;
  border-radius:36px;
  padding:48px;
  box-shadow:0 8px 24px rgba(0,0,0,.06);
  position: relative;
}
.loading-center {
  min-height:420px;
  display:flex;
  flex-direction:column;
  justify-content:center;
  align-items:center;
  text-align:center;
}
.loading-center .loader {
  width: 90px; height: 90px;
  border-radius:50%;
  border: 10px solid #e5e7eb;
  border-top-color:#2563eb;
  animation: spin 1s linear infinite;
  margin: 0 auto 12px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-center .stage-title {
  font-weight:900;
  font-size:22px;
  color:#111;
}
.loading-center .stage-desc  {
  font-weight:600;
  font-size:14px;
  color:#6b7280;
  margin-top:4px;
}

#elapsed_timer {
  position:absolute;
  right:20px;
  bottom:16px;
  font-weight:800;
  font-size:14px;
  color:#111;
  background:#fff;
  padding:8px 12px;
  border:2px dashed #111;
  border-radius:12px;
  z-index:5;
}

#result_wrap {
  background:#fff;
  border:3px dashed #111;
  border-radius:36px;
  padding:28px;
  box-shadow:0 8px 24px rgba(0,0,0,.06);
  position: relative;        
  padding-bottom: 92px;     
}
#res_layout {
  display:grid;
  grid-template-columns: 1.6fr 1fr;
  gap:28px;
  align-items:stretch;
}
#right_switch {
  position:relative;
  display:flex;
  flex-direction:column;
}

.qa-grid {
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:18px;
}
.qa-card {
  display:flex;
  align-items:center;
  justify-content:center;
  flex-direction:column;
  height:160px;
  border-radius:24px;
  background:#e5e7eb;
  color:#111;
  font-weight:800;
  white-space:pre-line;
  box-shadow:0 6px 14px rgba(0,0,0,.10);
  border:2px solid #d1d5db;
}
.qa-card .hash {
  margin-top:6px;
  font-weight:700;
  font-size:12px;
  color:#374151;
  opacity:.9;
}
.qa-card:hover {
  filter:brightness(1.03);
  transform:translateY(-1px);
  transition:.15s ease;
}

.detail-panel {
  background:#d1d5db;
  color:#111;
  border-radius:24px;
  padding:18px;
  box-shadow:0 10px 24px rgba(0,0,0,.18);
}
.detail-header {
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom:8px;
  gap:8px;
}
.detail-title {
  display:flex;
  align-items:center;
  gap:6px;
  font-weight:800;
  font-size:16px;
}
.detail-title .qa-tag {
  padding:3px 8px;
  border-radius:999px;
  font-size:10px;
  font-weight:700;
  background:rgba(17,24,39,.08);
  color:#111;
}

.detail-close,
button.detail-close,
.detail-header .gr-button.detail-close {
  width:18px !important;
  height:18px !important;
  min-width:18px !important;
  max-width:18px !important;
  padding:0 !important;
  margin:0 !important;
  border-radius:0 !important;
  font-size:12px !important;
  font-weight:700;
  line-height:1;
  display:flex !important;
  align-items:center;
  justify-content:center;
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
  color:#6b7280;
  cursor:pointer;
  overflow:hidden;
}

.detail-close:hover {
  color:#111;
  transform:scale(1.05);
}

/* Boxes inside detail */
.res-title {
  font-weight:800;
  font-size:18px;
  margin: 8px 0 6px;
}
.res-card {
  background:#f3f4f6;
  border:2px solid #cbd5e1;
  border-radius:12px;
  padding:12px 14px;
}

#home_btn_wrap{
  position:absolute;
  left:50%;
  transform:translateX(-50%);
  bottom:16px;
  z-index:20;
  width:auto;
  height:auto;
  padding:0;
  margin:0;
  border:none;
  background:transparent;
  pointer-events:none;
}
#home_btn_wrap *{ pointer-events:auto }
#home_btn{
  min-width:0 !important;
  width:44px;
  height:44px;
  padding:0;
  border-radius:999px;
  font-size:20px;
  font-weight:900;
  box-shadow:0 6px 16px rgba(0,0,0,.18);
}

#rag_row{
  display:grid !important;
  grid-template-columns: 1fr max-content;
  align-items:center;
  column-gap:12px;
  margin:4px 0 6px;
}
#rag_left{
  display:inline-flex;
  align-items:center;
  gap:8px;
  min-width:0;
  white-space:nowrap;
}
#rag_title{
  margin:0;
  line-height:1;
  white-space:nowrap;
}
#rag_toggle_ctrl{
  justify-self:end;
  width:auto !important;
  max-width:none !important;
}
#rag_toggle_ctrl > *{
  width:auto !important;
  max-width:none !important;
}

.rag-mini {
  display:inline-flex;
  align-items:center;
  gap:8px;
}
.rag-spinner {
  width:16px;
  height:16px;
  border-radius:50%;
  border:3px solid #e5e7eb;
  border-top-color:#4b5563;
  animation: spin .9s linear infinite;
}

#nav_bar, #step_indicator { display:none !important; }

.gradio-container .fixed.bottom-0.right-0,
.gradio-container .absolute.bottom-0.right-0,
.gradio-container [data-testid="status_tracker"],
.gradio-container [data-testid="queue-status"],
.gradio-container .gr-status-bar,
.gradio-container .resize-handle,
.gradio-container [role="status"].min-h-6,
.gradio-container [aria-live="polite"].min-h-6 { display:none !important; }
#loading_wrap .form { padding-bottom:0 !important; }
"""

TYPES = ["Factual", "Temporal", "Causal", "Counterfactual", "Predictive"]

# 타입별 기본값
DATA = {
    t: {
        "q": "",
        "r_internal": "",
        "r_external": "",
        "a_internal": "",
    }
    for t in TYPES
}

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_item(qtype: str, use_external: bool):
    item = DATA[qtype]
    q = item["q"]
    r = item["r_external"] if use_external else item["r_internal"]
    a = item["a_internal"]
    return q, r, a

def render_card(card_index: int, use_external: bool):
    idx = int(card_index) % len(TYPES)
    t = TYPES[idx]
    q, r, a = _get_item(t, use_external)
    rat_title_html = f"<div class='res-title'>{'Rationale with RAG' if use_external else 'Rationale'}</div>"
    step = f"{idx+1}/{len(TYPES)}"
    return "<div class='res-title'>Question</div>", q, rat_title_html, r, a, step


def open_detail(idx, card_index, _unused_toggle, states):
    i = int(idx)
    current_rag = (states or [False]*len(TYPES))[i % len(TYPES)]
    header, q, rtitle, r, a, step = render_card(i, current_rag)
    return (
        i, header, q, rtitle, r, a, step,
        gr.update(visible=False),          
        gr.update(visible=True),              
        gr.update(value=current_rag),        
    )

def close_detail():
    return gr.update(visible=True), gr.update(visible=False)

def on_file(files):
    if not files:
        return (
            gr.update(value=None, visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    return (
        gr.update(value=files, visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
    )
    
def _loading_markup(title: str, desc: str = ""):
    desc_html = f"<div class='stage-desc'>{desc}</div>" if desc else ""
    return f"""
    <div class="loading-center" aria-live="polite">
      <div class="loader" aria-hidden="true"></div>
      <div>
        <div class="stage-title">{title}</div>
        {desc_html}
      </div>
    </div>
    """

def show_loading_screen(vpath):
    if not vpath:
        return (
            gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(value="", visible=False), 0,
            gr.update(value="⏱ 0s", visible=False),
        )
    return (
        gr.update(visible=True), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False),
        gr.update(value=_loading_markup("QA Generating...", "Generating and filtering question-answer pairs."), visible=True),
        0, gr.update(value="⏱ 0s", visible=False),
    )

def hide_loading_screen():
    return (
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
    )

def set_rag_state(idx: int, states: list, val: bool):
    s = list(states) if isinstance(states, list) else [False]*len(TYPES)
    i = int(idx) % len(TYPES)
    s[i] = bool(val)
    return s

def rag_loading_inbox(use_external: bool):
    title = "Rationale with RAG" if use_external else "Rationale"
    return (
        gr.update(value=f"<div class='res-title'>{title}</div>"), 
        gr.update(value="⏳ Updating rationale...", interactive=False),
    )

def wait_short():
    time.sleep(1)
    return gr.update(visible=False)

def on_prev(i): return int(i) - 1
def on_next(i): return int(i) + 1

def on_home():
    step = f"1/{len(TYPES)}"
    return (
        gr.update(visible=False),               
        gr.update(visible=True),                
        gr.update(visible=False),                
        gr.update(value=None, visible=False),    
        gr.update(visible=False),               
        gr.update(value=None, visible=False),   
        gr.update(value=None),                  
        gr.update(value=False),                 
        0,                                      
        step,                                    
        "<div class='res-title'>Question</div>",
        "",                                      
        "<div class='res-title'>Rationale</div>", 
        "",                                      
        "",                              
        gr.update(visible=True),             
        gr.update(visible=False),                
        gr.update(value="", visible=False),    
        0,                                        
        gr.update(value="", visible=False),     
        [False] * len(TYPES),         
    )

def stage_qa(video_id: str, paths: dict):
    caption_data = _load_json(paths["CAPTION_INPUT"])
    if isinstance(caption_data, dict):
        videos_all = [{"VideoID": vid, **info} for vid, info in caption_data.items()]
    else:
        videos_all = caption_data

    videos = [v for v in videos_all
              if str(v.get("VideoID") or v.get("video_id")) == str(video_id)]
    if not videos:
        raise ValueError(f"[stage_qa] video_id '{video_id}' not found in caption data.")

    qa_data = []
    for item in tqdm(videos, desc=f"QA 생성 ({video_id})"):
        vid = item.get("VideoID") or item.get("video_id")
        caps = (
            item.get("OriginalCaptions")
            or item.get("original_captions")
            or item.get("sentences")  
        )
        ts = item.get("timestamps", [])

        if not caps:
            print(f"[stage_qa] video_id={vid} 캡션이 없어서 건너뜀")
            continue
         
        res = generate_reasoning_problems_as_is(
            captions=caps,
            video_id=vid,
            timestamps=ts,
            api_key=OPENAI_API_KEY,
            model=GPT_MODEL,
        )
        if not res: 
            continue
        if isinstance(res, list):
            qa_data.extend(res)
        else:
            qa_data.append(res)

    save_json(qa_data, paths["QA_OUTPUT"])

    cfg_qa = FilterConfig(drop_rejects=False)
    qa_filtered = run_filter_on_memory_data(qa_data, cfg_qa)
    save_json(qa_filtered, paths["QA_FILTER_OUTPUT"])
    return qa_filtered

def stage_keyframe(video_id: str, paths: dict, qa_filtered):
    ensure_frames_for_video(
        video_id=video_id,
        frame_root=paths["FRAME_ROOT"],
        video_dir=VIDEO_SAVE_DIR,                # demo에서 이미 쓰고 있는 업로드 폴더
        fps=int(FPS) if isinstance(FPS, (int, float)) else 1,
        overwrite=True,
    )

    model, tokenizer, processor = setup_model()
    frame_scored = generate_frame_scores_per_video(
        videos_to_process=qa_filtered,
        existing_results=[],
        frame_root=paths["FRAME_ROOT"],
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        output_path=paths["SCORE_OUTPUT"],
        fps=FPS,
        chunk_size=CHUNK_SIZE,
        save_interval=SAVE_INTERVAL,
    )

    save_json(frame_scored, paths["SCORE_OUTPUT"])
    return frame_scored

def stage_rationale(video_id: str, paths: dict, frame_scored):
    rationale_data = []
    for video in tqdm(frame_scored, desc=f"Rationale 생성 ({video_id})"):
        new_video = {
            "VideoID": video["VideoID"],
            "OriginalCaptions": video["OriginalCaptions"],
            "Problem-AnswerPairs": [],
        }
        for qa_pair in video.get("Problem-AnswerPairs", []):
            updated = generate_rationale_for_qa(
                video_qa=video,
                qa_pair=qa_pair,
                api_key=OPENAI_API_KEY,
                model=GPT_MODEL,
                temperature=RATIONALE_TEMPERATURE,
                max_tokens=RATIONALE_MAX_TOKENS,
            )
            new_video["Problem-AnswerPairs"].append(updated)
        rationale_data.append(new_video)

    save_json(rationale_data, paths["RATIONALE_OUTPUT"])

    scored_rows, rationale_validated = score_and_filter(
        rationale_data,
        cfg=RoscoeConfig(
            drop_failures=False,
            save_scores_json=paths["ROSCOE_SCORES"],
        ),
        th=RoscoeThresholds(),
    )
    save_json(rationale_validated, paths["RATIONALE_VALIDATED"])
    return rationale_validated

def stage_rag(video_id: str, paths: dict, rationale_validated):
    rag_client, rag_collection, rag_model = setup_rag_models(
        db_directory=CHROMA_DB_DIRECTORY,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    final_rag = run_rag_pipeline(
        rationale_data=rationale_validated,
        rag_client=rag_client,
        rag_collection=rag_collection,
        rag_model=rag_model,
        api_key=OPENAI_API_KEY,
        gpt_model=GPT_MODEL,
    )
    save_json(final_rag, paths["RAG_OUTPUT"])
    return final_rag

def stage_qa_ui(vpath, elapsed: int):
    start = time.time()

    fname = os.path.basename(vpath)
    dst = os.path.join(VIDEO_SAVE_DIR, fname)
    if not os.path.exists(dst):
        shutil.copy2(vpath, dst)
    video_id = os.path.splitext(fname)[0]

    file_name = f"{DEFAULT_FILE_NAME}_{video_id}"
    paths = get_file_paths(file_name)
    base_paths = get_file_paths(DEFAULT_FILE_NAME)
    paths["CAPTION_INPUT"] = base_paths["CAPTION_INPUT"]
    paths["FRAME_ROOT"]     = base_paths["FRAME_ROOT"]

    qa_filtered = stage_qa(video_id, paths)

    elapsed += int(time.time() - start)
    html = _loading_markup("Keyframes Selecting...", "Selecting important frames and calculating scores.")

    return (
        gr.update(value=html, visible=True),
        elapsed,
        video_id,
        paths,
    )

def stage_keyframe_ui(video_id: str, paths: dict, elapsed: int):
    start = time.time()
    qa_filtered = _load_json(paths["QA_FILTER_OUTPUT"])
    frame_scored = stage_keyframe(video_id, paths, qa_filtered)

    elapsed += int(time.time() - start)
    html = _loading_markup("Rationale Generating...", "Generating and validating logical rationale for each question.")
    return (
        gr.update(value=html, visible=True),
        elapsed,
    )


def stage_rationale_ui(video_id: str, paths: dict, elapsed: int):
    start = time.time()
    frame_scored = _load_json(paths["SCORE_OUTPUT"])
    rationale_validated = stage_rationale(video_id, paths, frame_scored)

    elapsed += int(time.time() - start)
    html = _loading_markup("Rationale Reinforcing...", "Reinforcing the Rationale process with external knowledge")
    return (
        gr.update(value=html, visible=True),
        elapsed,
    )


def stage_rag_ui(video_id: str, paths: dict, use_external: bool, vpath: str):
    rationale_validated = _load_json(paths["RATIONALE_VALIDATED"])
    final_rag = stage_rag(video_id, paths, rationale_validated)

    global DATA
    for t in TYPES:
        DATA[t] = {"q": "", "r_internal": "", "r_external": "", "a_internal": ""}

    v_rat = rationale_validated[0]
    v_rag = final_rag[0]

    pairs_rat = v_rat.get("Problem-AnswerPairs", [])
    pairs_rag = v_rag.get("Problem-AnswerPairs", [])

    by_dim_rat = {p.get("Dimension"): p for p in pairs_rat}
    by_dim_rag = {p.get("Dimension"): p for p in pairs_rag}

    for dim in TYPES:
        p_in = by_dim_rat.get(dim)
        if not p_in: 
            continue
        p_ex = by_dim_rag.get(dim, p_in)

        q_text = p_in.get("Question", "")
        ans_in = p_in.get("Answer", {})
        ans_ex = p_ex.get("Answer", {})

        if isinstance(ans_in, dict):
            a_internal = ans_in.get("Answer", "")
            r_internal = ans_in.get("Rationale", "")
        else:
            a_internal = str(ans_in)
            r_internal = ""

        if isinstance(ans_ex, dict):
            r_external = ans_ex.get("Rationale", "") or r_internal
        else:
            r_external = r_internal

        DATA[dim] = {
            "q": q_text,
            "r_internal": r_internal,
            "r_external": r_external,
            "a_internal": a_internal,
        }

    header, q, rat_title_html, r, a, step = render_card(0, use_external)

    return (
        gr.update(visible=True),     
        gr.update(visible=False),        
        gr.update(visible=False),         
        gr.update(value=vpath, visible=True), 
        0,                                
        step,
        header, q, rat_title_html, r, a,
        gr.update(value=use_external),
    )

# ---- UI ----
with gr.Blocks(title="Video QA", css=CSS, theme=gr.themes.Soft(primary_hue="slate", neutral_hue="gray")) as demo:
    with gr.Column(elem_id="page_wrap"):
        # Upload
        with gr.Column(elem_id="upload_section") as upload_section:
            dropzone_html = gr.HTML("""
            <div id="dropzone">
              <div class="inner">
                <svg viewBox="0 0 64 64" aria-hidden="true">
                  <path d="M19 40c-6 0-11-5-11-11s5-11 11-11c1 0 2 .1 3 .4C24 14 28 11 33 11c7 0 12 6 12 13v1h1c5 0 9 4 9 9s-4 9-9 9h-9"/>
                  <path d="M32 46V26M24 34l8-8 8 8"/>
                </svg>
                <div class="label">Video Upload</div>
              </div>
            </div>
            """)
            file_overlay = gr.File(label=None, file_count="single",
                                   file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm"],
                                   elem_id="file_overlay")

        with gr.Column(visible=False, elem_id="preview_box") as preview_box:
            video_preview = gr.Video(label=None, show_label=False, container=False,
                                     elem_id="video_comp", visible=False, autoplay=False,
                                     height=520, sources=["upload"])
            gen_btn = gr.Button("Generate", variant="primary", elem_id="gen_btn", visible=False)

        with gr.Column(visible=False, elem_id="loading_wrap") as loading_wrap:
            loading_html = gr.HTML(value="", visible=False)
            elapsed_timer = gr.HTML(value="⏱ 0s", elem_id="elapsed_timer", visible=False)

        with gr.Column(visible=False, elem_id="result_wrap") as result_wrap:
            with gr.Row(elem_id="res_layout"):
                # Left: video
                result_video = gr.Video(label=None, show_label=False, container=False,
                                        elem_id="video_comp", autoplay=False, height=520, sources=["upload"])
                
                with gr.Column(elem_id="right_switch") as right_switch:
                    with gr.Column(visible=True) as qa_grid:
                        with gr.Row(elem_classes=["qa-grid"]):
                            btn_temporal = gr.Button("Q&A\n#Temporal", elem_classes=["qa-card"])
                            btn_causal   = gr.Button("Q&A\n#Causal",   elem_classes=["qa-card"])
                            btn_counter  = gr.Button("Q&A\n#Counterfactual", elem_classes=["qa-card"])
                            btn_factual  = gr.Button("Q&A\n#Factual",  elem_classes=["qa-card"])
                    with gr.Column(visible=False, elem_classes=["detail-panel"]) as detail_panel:
                        with gr.Row(elem_classes=["detail-header"]):
                            detail_title = gr.HTML("<div class='detail-title'>Q&A</div>")
                            detail_close = gr.Button("✕", elem_classes=["detail-close"], variant="secondary")
                        q_title = gr.HTML("<div class='res-title'>Question</div>")
                        q_out   = gr.Textbox(value="", lines=3, interactive=False, container=False, elem_classes=["res-card"])
                        with gr.Row(elem_id="rag_row"):
                            rat_title  = gr.HTML("<div id='rag_left'><span id='rag_title' class='res-title'>Rationale</span></div>")
                            rag_toggle  = gr.Checkbox(value=False, label="", show_label=False, container=False, elem_id="rag_toggle_ctrl")
                        rag_loading = gr.HTML(value="", visible=False)  # mini loader slot
                        r_out = gr.Textbox(value="", lines=4, interactive=False, container=False, elem_classes=["res-card"])
                        gr.HTML("<div class='res-title'>Answer</div>")
                        a_out = gr.Textbox(value="", lines=2, interactive=False, container=False, elem_classes=["res-card"])

            with gr.Row(elem_id="home_btn_wrap"):
                home = gr.Button("🏠", elem_id="home_btn", variant="secondary")

            with gr.Row(elem_id="nav_bar"):
                prev_btn = gr.Button("◀ 이전", elem_id="prev_btn", variant="secondary", visible=False)
                _dummy_home = gr.Button("🏠", visible=False)
                next_btn = gr.Button("다음 ▶", elem_id="next_btn", variant="secondary", visible=False)
            step_label = gr.HTML("<div id='step_indicator'>1/5</div>", visible=False)

        card_index = gr.State(0)
        elapsed_state = gr.State(0)
        rag_states = gr.State([False]*len(TYPES)) 
        video_id_state = gr.State("")             
        paths_state = gr.State({})         

    file_overlay.change(
        on_file, inputs=file_overlay,
        outputs=[video_preview, dropzone_html, preview_box, gen_btn],
        show_progress=False,
    ).then(
        lambda vp: gr.update(visible=True) if vp is not None else gr.update(visible=False),
        inputs=video_preview, outputs=preview_box,
        show_progress=False,
    )

    gen_btn.click(
        show_loading_screen,
        inputs=[video_preview],
        outputs=[loading_wrap, upload_section, preview_box, result_wrap,
                loading_html, elapsed_state, elapsed_timer],
        show_progress=False,
    ).then(
        stage_qa_ui,
        inputs=[video_preview, elapsed_state],
        outputs=[loading_html, elapsed_state, video_id_state, paths_state],
        show_progress=False,
    ).then(
        stage_keyframe_ui,
        inputs=[video_id_state, paths_state, elapsed_state],
        outputs=[loading_html, elapsed_state],
        show_progress=False,
    ).then(
        stage_rationale_ui,
        inputs=[video_id_state, paths_state, elapsed_state],
        outputs=[loading_html, elapsed_state],
        show_progress=False,
    ).then(
        stage_rag_ui,
        inputs=[video_id_state, paths_state, rag_toggle, video_preview],
        outputs=[result_wrap, upload_section, preview_box, result_video,
                card_index, step_label, q_title, q_out, rat_title, r_out, a_out, rag_toggle],
        show_progress=False,
    ).then(
        hide_loading_screen,
        outputs=[loading_wrap, loading_html, elapsed_timer],
        show_progress=False,
    )

    prev_btn.click(on_prev, inputs=[card_index], outputs=[card_index], show_progress=False).then(
        render_card, inputs=[card_index, rag_toggle],
        outputs=[q_title, q_out, rat_title, r_out, a_out, step_label],
        show_progress=False
    )
    next_btn.click(on_next, inputs=[card_index], outputs=[card_index], show_progress=False).then(
        render_card, inputs=[card_index, rag_toggle],
        outputs=[q_title, q_out, rat_title, r_out, a_out, step_label],
        show_progress=False
    )

    rag_toggle.change(
        set_rag_state,
        inputs=[card_index, rag_states, rag_toggle],
        outputs=[rag_states],                 
        show_progress=False
    ).then(
        rag_loading_inbox,
        inputs=[rag_toggle],                   
        outputs=[rat_title, r_out],
        show_progress=False
    ).then(
        wait_short,
        show_progress=False
    ).then(
        render_card,
        inputs=[card_index, rag_toggle],       
        outputs=[q_title, q_out, rat_title, r_out, a_out, step_label],
        show_progress=False
    )

    btn_temporal.click(
        open_detail, inputs=[gr.State(1), card_index, rag_toggle, rag_states],
        outputs=[card_index, q_title, q_out, rat_title, r_out, a_out, step_label, qa_grid, detail_panel, rag_toggle],
        show_progress=False
    )
    btn_causal.click(
        open_detail, inputs=[gr.State(2), card_index, rag_toggle, rag_states],
        outputs=[card_index, q_title, q_out, rat_title, r_out, a_out, step_label, qa_grid, detail_panel, rag_toggle],
        show_progress=False
    )
    btn_counter.click(
        open_detail, inputs=[gr.State(3), card_index, rag_toggle, rag_states],
        outputs=[card_index, q_title, q_out, rat_title, r_out, a_out, step_label, qa_grid, detail_panel, rag_toggle],
        show_progress=False
    )
    btn_factual.click(
        open_detail, inputs=[gr.State(0), card_index, rag_toggle, rag_states],
        outputs=[card_index, q_title, q_out, rat_title, r_out, a_out, step_label, qa_grid, detail_panel, rag_toggle],
        show_progress=False
    )
    detail_close.click(close_detail, outputs=[qa_grid, detail_panel], show_progress=False)

    home.click(
        on_home,
        outputs=[
            result_wrap, upload_section, preview_box, video_preview, gen_btn, result_video,
            file_overlay, rag_toggle, card_index, step_label, q_title, q_out, rat_title, r_out, a_out,
            dropzone_html, loading_wrap, loading_html, elapsed_state, elapsed_timer, rag_states
        ],
        show_progress=False
    )

if __name__ == "__main__":
    demo.launch()