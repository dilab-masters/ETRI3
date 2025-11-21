import os
import cv2
import subprocess
from tqdm import tqdm
import shutil   # ← 추가

def convert_to_h264(input_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except subprocess.CalledProcessError:
        print(f"Error: Failed to convert {input_path}")
        return None

def video_to_frames(video_path: str, output_dir: str, fps: int = 1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    tmp_path = None
    if not cap.isOpened():
        tmp_path = video_path.replace(".mp4", "_h264.mp4")
        print(f"Cannot open {video_path}, converting to H.264...")
        converted = convert_to_h264(video_path, tmp_path)
        if converted:
            cap = cv2.VideoCapture(converted)
        else:
            return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    frame_paths = []
    sec = 0
    while sec < duration:
        frame_index = int(sec * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"sec{int(sec):05d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        sec += 1 / fps

    cap.release()

    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)
        print(f"Deleted temporary file {tmp_path}")

    print(f"Extracted {len(frame_paths)} frames from {video_path} to {output_dir}")
    return frame_paths

def ensure_frames_for_video(
    video_id: str,
    frame_root: str,
    video_dir: str,
    fps: int = 1,
    overwrite: bool = True,
) -> str:
    """
    - video_dir 안에서 stem == video_id 인 비디오 파일을 찾아서
    - frame_root/video_id 아래에 프레임을 추출한다.
    - overwrite=True 이면 기존 프레임 폴더를 지우고 새로 만든다.
    - 반환값: 실제 프레임이 저장된 디렉토리 경로
    """
    # 1) 프레임 저장 디렉토리
    frame_dir = os.path.join(frame_root, video_id)

    if overwrite and os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    # 2) 업로드 비디오 파일 찾기
    vid_path = None
    if os.path.isdir(video_dir):
        for name in os.listdir(video_dir):
            stem, ext = os.path.splitext(name)
            if stem == str(video_id) and ext.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                vid_path = os.path.join(video_dir, name)
                break

    if vid_path is None:
        print(f"[ensure_frames_for_video] video '{video_id}' not found in {video_dir}")
        return frame_dir

    # 3) 실제 프레임 추출
    try:
        use_fps = int(fps) if isinstance(fps, (int, float)) and fps > 0 else 1
        print(f"[INFO] Extracting frames for {video_id} from {vid_path} -> {frame_dir} (fps={use_fps})")
        video_to_frames(vid_path, frame_dir, fps=use_fps)
    except Exception as e:
        print(f"[WARN] Failed to extract frames for {video_id}: {e}")

    return frame_dir

if __name__ == "__main__":
    video_folder = "."
    frame_base_folder = "./Data/Frame"

    for file in os.listdir(video_folder):
        if file.endswith(".mp4"):
            video_path = os.path.join(video_folder, file)
            video_id = os.path.splitext(file)[0]
            output_dir = os.path.join(frame_base_folder, video_id)

            if os.path.exists(output_dir):
                print(f"Skipping {video_id}, frames already extracted in {output_dir}")
                continue

            video_to_frames(video_path, output_dir, fps=1)