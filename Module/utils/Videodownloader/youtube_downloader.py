#!/usr/bin/env python3

"""
Need to install yt-dlp and ffmpeg.
python youtube_downloader.py --json input.json --out /path/to/output
"""

import os, json, argparse, shutil, csv
from typing import Iterable, Set, Optional
from yt_dlp import YoutubeDL

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not installed. Install ffmpeg using your package manager.")

def extract_yt_id_from_token(tok: str) -> Optional[str]:
    if not tok or not isinstance(tok, str):
        return None
    t = tok.strip()

    if t.startswith("v_") and len(t) > 2:
        return t[2:]

    parts = t.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])

    return t or None

def iter_ids_from_json_obj(obj) -> Iterable[str]:
    if isinstance(obj, dict):
        for k in obj.keys():
            yt_id = extract_yt_id_from_token(k)
            if yt_id:
                yield yt_id

        for v in obj.values():
            if isinstance(v, dict):
                for key in ("VideoID", "video_id"):
                    if key in v and isinstance(v[key], str):
                        yt_id = extract_yt_id_from_token(v[key])
                        if yt_id:
                            yield yt_id
            elif isinstance(v, list):
                for rec in v:
                    if isinstance(rec, dict):
                        for key in ("VideoID", "video_id"):
                            if key in rec and isinstance(rec[key], str):
                                yt_id = extract_yt_id_from_token(rec[key])
                                if yt_id:
                                    yield yt_id

    elif isinstance(obj, list):
        for rec in obj:
            if isinstance(rec, dict):
                for k in rec.keys():
                    yt_id = extract_yt_id_from_token(k)
                    if yt_id:
                        yield yt_id

def iter_yt_ids(json_path: str) -> Iterable[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        if first.lstrip().startswith("[") or first.lstrip().startswith("{"):
            buf = first + f.read()
            try:
                obj = json.loads(buf)
            except json.JSONDecodeError:
                pass
            else:
                yield from iter_ids_from_json_obj(obj)
                return
        try:
            rec = json.loads(first)
            yield from iter_ids_from_json_obj(rec)
        except json.JSONDecodeError:
            pass
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield from iter_ids_from_json_obj(rec)

def download_full_video(
    yt_id: str,
    out_root: str,
    overwrite: bool = False,
    cookies_from_browser: Optional[str] = None,
    cookie_file: Optional[str] = None,
) -> None:
    os.makedirs(out_root, exist_ok=True)

    save_id = yt_id if yt_id.startswith("v_") else f"v_{yt_id}"
    out_file_tpl = os.path.join(out_root, f"{save_id}.%(ext)s")

    if not overwrite:
        base_prefix = os.path.join(out_root, f"{save_id}.")
        for ext in ("mp4", "mkv", "webm", "mov", "m4v"):
            if os.path.exists(base_prefix + ext):
                print(f"[skip] exists: {save_id}")
                return

    real_id = yt_id[2:] if yt_id.startswith("v_") else yt_id
    url = f"https://www.youtube.com/watch?v={real_id}"

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
    ap.add_argument("--json", required=True, help="JSON/JSONL file path")
    ap.add_argument("--out", default="./videos", help="Output directory")
    ap.add_argument("--limit", type=int, default=None, help="(Optional) Maximum number of items to process")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite files with same name")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="Example: chrome, firefox, chromium, edge (for login-required videos)")
    ap.add_argument("--cookie-file", default=None,
                    help="Netscape format cookies.txt file path")
    ap.add_argument("--fail-log", default="failed.csv",
                    help="CSV path for logging failed items")
    args = ap.parse_args()

    ensure_ffmpeg()

    uniq: Set[str] = set()
    for yt_id in iter_yt_ids(args.json):
        if yt_id:
            uniq.add(yt_id)

    print(f"[info] unique YouTube IDs: {len(uniq)}")

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
            print(f"[error] Failed: {yt_id} ({msg})")
            if "Private video" in msg or "Sign in" in msg:
                etype = "auth_required"
            elif "unavailable" in msg.lower():
                etype = "unavailable"
            else:
                etype = "other"
            fail_rows.append([yt_id, etype, msg])
        else:
            print(f"[done] {yt_id}")

        count += 1
        if args.limit and count >= args.limit:
            print(f"[info] Reached limit {args.limit}, stopping")
            break

    if fail_rows:
        fail_path = os.path.abspath(args.fail_log)
        os.makedirs(os.path.dirname(fail_path) or ".", exist_ok=True)
        import csv
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["yt_id", "error_type", "message"])
            w.writerows(fail_rows)
        print(f"[info] Logged {len(fail_rows)} failures → {fail_path}")

if __name__ == "__main__":
    main()