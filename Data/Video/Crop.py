import os
import cv2
import subprocess
from tqdm import tqdm

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