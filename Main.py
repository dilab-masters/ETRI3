import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI

from config import (
    get_file_paths,
    OPENAI_API_KEY,
    GPT_MODEL,
    RATIONALE_TEMPERATURE,
    RATIONALE_MAX_TOKENS,
    FPS,
    CHUNK_SIZE,
    SAVE_INTERVAL,
    FRAME_ROOT,
    DEFAULT_FILE_NAME,
    QWEN_MODEL_NAME,
    CHROMA_DB_DIRECTORY,
    CHROMA_COLLECTION_NAME,
)

from Module.QA_generation import generate_reasoning_problems_as_is
from Module.Rationale_generation import generate_rationale_for_qa
from Module.Rationale_select import generate_frame_scores_per_video, save_results
from Module.utils.Key_Frame import setup_model

from Module.QA_filtering import (
    FilterConfig,
    run_filter_on_memory_data,
    save_json,
    _hard_drop_rejects,
)

from Module.Rationale_filtering import (
    RoscoeThresholds,
    RoscoeConfig,
    score_and_filter,
)

from Module.RAG_integration import setup_rag_models, run_rag_pipeline


def load_json_data(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(file_name: str):
    paths = get_file_paths(file_name)
    print(f"\n{'='*60}")
    print(f"Starting data generation pipeline for '{file_name}'")
    print(f"{'='*60}\n")

    print("[Step 1/5] Generating Question-Answer (QA) pairs...")
    caption_data = load_json_data(paths["CAPTION_INPUT"])

    if isinstance(caption_data, dict):
        caption_data = [
            {"video_id": vid, **info} for vid, info in caption_data.items()
        ]

    qa_results = []
    progress_bar = tqdm(caption_data, desc="Generating QA")

    for video_info in progress_bar:
        generated_qa = generate_reasoning_problems_as_is(
            captions=video_info["sentences"],
            video_id=video_info["video_id"],
            timestamps=video_info["timestamps"],
            api_key=OPENAI_API_KEY,
            model=GPT_MODEL,
        )
        if generated_qa:
            qa_results.append(generated_qa)

    save_results(qa_results, paths["QA_OUTPUT"])
    qa_data = qa_results
    print(f"QA generation completed\n")

    print("[Step 2/5] QA Quality Filtering (Qwen-based)...")
    if os.path.exists(paths["QA_FILTER_OUTPUT"]):
        print(f"Already exists: {paths['QA_FILTER_OUTPUT']}")
        qa_filtered_data = load_json_data(paths["QA_FILTER_OUTPUT"])
    else:
        flt_cfg = FilterConfig(
            model_name=QWEN_MODEL_NAME,
            cap_max_captions=0,
            cap_max_chars=6000,
            cap_use_referenced_only=True,
            max_new_tokens=192,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            drop_rejects=True,
            device_map="auto",
        )
        qa_filtered_data = run_filter_on_memory_data(qa_data, flt_cfg)
        save_json(qa_filtered_data, paths["QA_FILTER_OUTPUT"])

    qa_filtered_data = _hard_drop_rejects(qa_filtered_data)
    print(f"QA filtering completed\n")

    print("[Step 2/5] Visual Evidence Discovery (Frame Scoring)...")
    if os.path.exists(paths["SCORE_OUTPUT"]):
        print(f"Already exists: {paths['SCORE_OUTPUT']}")
        scored_data = load_json_data(paths["SCORE_OUTPUT"])
    else:
        print("Loading Vision-Language model...")
        model, tokenizer, processor = setup_model()

        scored_data = generate_frame_scores_per_video(
            videos_to_process=qa_filtered_data,
            existing_results=[],
            frame_root=FRAME_ROOT,
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            output_path=paths["SCORE_OUTPUT"],
            fps=FPS,
            chunk_size=CHUNK_SIZE,
            save_interval=SAVE_INTERVAL,
        )

        save_results(scored_data, paths["SCORE_OUTPUT"])
    print(f"Frame scoring completed\n")

    print("[Step 3/5] Logical Evidence (Rationale) Generation...")
    if os.path.exists(paths["RATIONALE_OUTPUT"]):
        print(f"Already exists: {paths['RATIONALE_OUTPUT']}")
        rationale_data = load_json_data(paths["RATIONALE_OUTPUT"])
    else:
        rationale_results = []
        progress_bar = tqdm(scored_data, desc="Generating Rationale")

        for video_qa in progress_bar:
            processed_pairs = []

            for pair in video_qa.get("Problem-AnswerPairs", []):
                updated_pair = generate_rationale_for_qa(
                    video_qa=video_qa,
                    qa_pair=pair,
                    api_key=OPENAI_API_KEY,
                    model=GPT_MODEL,
                    temperature=RATIONALE_TEMPERATURE,
                    max_tokens=RATIONALE_MAX_TOKENS,
                )
                if updated_pair:
                    processed_pairs.append(updated_pair)

            video_qa["Problem-AnswerPairs"] = processed_pairs
            rationale_results.append(video_qa)

        save_results(rationale_results, paths["RATIONALE_OUTPUT"])
        rationale_data = rationale_results
    print(f"Rationale generation completed\n")

    print("[Step 3.5] ROSCOE-based Rationale Quality Validation...")
    rationale_data = load_json_data(paths["RATIONALE_OUTPUT"])

    cfg = RoscoeConfig(
        cap_max=0,
        drop_failures=True,
        save_scores_json=None,
        device=None,
    )

    th = RoscoeThresholds(
        SA_Faith_step=0.65,
        SS_Info_chain=0.65,
        SS_Rep_step=0.02,
        LI_Self_cons=0.50,
        LC_PPL_inv=0.02,
        LC_Grammar=0.65,
        d_SA_Faith=0.0,
        d_SS_Info=0.0,
        allow_none=True,
        require_all=True,
    )

    scored_rows, validated_data = score_and_filter(rationale_data, cfg=cfg, th=th)

    save_results(scored_rows, paths["ROSCOE_SCORES"])
    save_results(validated_data, paths["RATIONALE_VALIDATED"])
    print(f"Rationale validation completed\n")

    print("[Step 4/5] External Knowledge Integration (RAG)...")
    if os.path.exists(paths["RAG_OUTPUT"]):
        print(f"Already exists: {paths['RAG_OUTPUT']}")
    else:
        if not os.path.exists(paths["RAG_INPUT"]):
            raise FileNotFoundError(
                f"RAG input file not found: {paths['RAG_INPUT']}\n"
                f"Ensure Step 3.5 (RATIONALE_VALIDATED) completed successfully."
            )
        rag_input_data = load_json_data(paths["RAG_INPUT"])

        print("Loading RAG models (ChromaDB, BAAI)...")
        rag_client, rag_collection, rag_model = setup_rag_models(
            CHROMA_DB_DIRECTORY,
            CHROMA_COLLECTION_NAME,
        )

        if not rag_client or not rag_collection or not rag_model:
            print("⚠ RAG setup failed: ChromaDB or embedding model initialization error. Skipping step.")
            return

        final_rag_data = run_rag_pipeline(
            rationale_data=rag_input_data,
            rag_client=rag_client,
            rag_collection=rag_collection,
            rag_model=rag_model,
            api_key=OPENAI_API_KEY,
            gpt_model=GPT_MODEL,
        )

        save_results(final_rag_data, paths["RAG_OUTPUT"])
    print(f"RAG integration completed\n")

    print("[Step 5/5] Finalizing results...")
    print(f"{'='*60}")
    print(f"Pipeline completed successfully!")
    print(f"Final output: {paths['RAG_OUTPUT']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete video QA data generation pipeline."
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=DEFAULT_FILE_NAME,
        help=f"Caption file name to process (without extension). Default: '{DEFAULT_FILE_NAME}'",
    )
    args = parser.parse_args()

    run_pipeline(args.file_name)