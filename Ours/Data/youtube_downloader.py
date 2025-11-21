#!/usr/bin/env python3
"""
python youtube_downloader.py --json ./Ours/Data_QA/QA_Ours.json --out ./Video_Ours
"""
import os
import json
import argparse
import shutil
import csv
from typing import Set, Optional
from yt_dlp import YoutubeDL

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not installed. Install ffmpeg using your package manager.")

def extract_yt_id_from_video_id(video_id: str) -> Optional[str]:
    if not video_id or not isinstance(video_id, str):
        return None
    
    video_id = video_id.strip()
    
    if video_id.startswith("v_"):
        return video_id[2:]
    
    return video_id

def extract_unique_video_ids(json_path: str) -> Set[str]:
    unique_ids = set()
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                video_id = item.get("VideoID")
                if video_id:
                    yt_id = extract_yt_id_from_video_id(video_id)
                    if yt_id:
                        unique_ids.add(yt_id)
    
    return unique_ids

def download_full_video(
    yt_id: str,
    out_root: str,
    overwrite: bool = False,
    cookies_from_browser: Optional[str] = None,
    cookie_file: Optional[str] = None,
) -> None:
    os.makedirs(out_root, exist_ok=True)

    save_id = f"v_{yt_id}"
    out_file_tpl = os.path.join(out_root, f"{save_id}.%(ext)s")

    if not overwrite:
        base_prefix = os.path.join(out_root, f"{save_id}.")
        for ext in ("mp4", "mkv", "webm", "mov", "m4v"):
            if os.path.exists(base_prefix + ext):
                print(f"[skip] exists: {save_id}")
                return

    url = f"https://www.youtube.com/watch?v={yt_id}"

    ydl_opts = {
        "outtmpl": out_file_tpl,
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
        "retries": 3,
        "fragment_retries": 3,
        "overwrites": bool(overwrite),
    }

    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = cookies_from_browser
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(f"[done] {save_id}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/home/dilab/Desktop/QAGen/Ours/Data/QA_Ours.json", 
                    help="JSON file path with VideoID field")
    ap.add_argument("--out", default="./videos", help="Output directory")
    ap.add_argument("--limit", type=int, default=None, help="Maximum number of videos to download")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="Browser to extract cookies from (chrome, firefox, chromium, edge)")
    ap.add_argument("--cookie-file", default=None,
                    help="Path to cookies.txt file in Netscape format")
    ap.add_argument("--fail-log", default="failed.csv",
                    help="CSV file path for logging failed downloads")
    args = ap.parse_args()

    ensure_ffmpeg()

    uniq = extract_unique_video_ids(args.json)
    
    print(f"[info] Found {len(uniq)} unique YouTube IDs from {args.json}")
    print(f"[info] Starting download to {os.path.abspath(args.out)}")

    fail_rows = []
    count = 0

    for yt_id in sorted(uniq):
        try:
            download_full_video(
                yt_id=yt_id,
                out_root=args.out,
                overwrite=args.overwrite,
                cookies_from_browser=args.cookies_from_browser,
                cookie_file=args.cookie_file,
            )
        except Exception as e:
            msg = str(e)
            print(f"[error] Failed: v_{yt_id} ({msg})")
            if "Private video" in msg or "Sign in" in msg:
                etype = "auth_required"
            elif "unavailable" in msg.lower():
                etype = "unavailable"
            else:
                etype = "other"
            fail_rows.append([yt_id, etype, msg])
        else:
            print(f"[done] v_{yt_id}")

        count += 1
        if args.limit and count >= args.limit:
            print(f"[info] Reached limit {args.limit}, stopping")
            break

    if fail_rows:
        fail_path = os.path.abspath(args.fail_log)
        os.makedirs(os.path.dirname(fail_path) or ".", exist_ok=True)
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["yt_id", "error_type", "message"])
            w.writerows(fail_rows)
        print(f"[info] Logged {len(fail_rows)} failures → {fail_path}")

    success_count = count - len(fail_rows)
    print(f"\n[summary] Downloaded: {success_count}/{count} videos")

if __name__ == "__main__":
    main()