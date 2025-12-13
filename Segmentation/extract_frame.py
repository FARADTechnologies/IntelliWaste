import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2


@dataclass
class FrameRequest:
    frame_index: Optional[int] = None
    timestamp_sec: Optional[float] = None

    def resolve_frame_index(self, fps: float) -> int:
        if fps <= 0:
            raise RuntimeError(f"Invalid FPS reported by video reader: {fps}")

        if self.frame_index is not None and self.timestamp_sec is not None:
            raise ValueError("Provide either frame_index or timestamp_sec, not both.")
        if self.frame_index is None and self.timestamp_sec is None:
            return 0  # default to first frame

        if self.frame_index is not None:
            if self.frame_index < 0:
                raise ValueError("frame_index must be >= 0")
            return self.frame_index

        assert self.timestamp_sec is not None
        if self.timestamp_sec < 0:
            raise ValueError("timestamp_sec must be >= 0")
        return int(round(self.timestamp_sec * fps))


def _open_video(video_path: Path) -> Tuple[cv2.VideoCapture, float, int]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(
            f"Failed to read video metadata (fps={fps}, frames={frame_count}) from: {video_path}"
        )
    return cap, fps, frame_count


def extract_frame(video_path: Path, output_path: Path, request: FrameRequest) -> Path:
    """
    Read a specific frame from a video file and save it as an image.

    Raises descriptive exceptions when the video cannot be opened or read.
    """
    cap, fps, frame_count = _open_video(video_path)
    target_index = request.resolve_frame_index(fps=fps)

    if target_index >= frame_count:
        cap.release()
        raise IndexError(
            f"Requested frame_index {target_index} exceeds total frames ({frame_count}). "
            f"Try a smaller timestamp/frame."
        )

    # Seek to the target frame
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, target_index):
        cap.release()
        raise RuntimeError(f"Failed to seek to frame {target_index} in {video_path}")

    success, frame = cap.read()
    cap.release()
    if not success or frame is None:
        raise RuntimeError(f"Failed to read frame {target_index} from: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to save frame to: {output_path}")

    return output_path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    video_path = repo_root / "dumpster01.mp4"

    # Default: grab a frame around 3s (≈90th frame at 30 FPS)
    request = FrameRequest(timestamp_sec=3.0)
    output_path = repo_root / "frame_at_3s.jpg"

    saved_path = extract_frame(video_path, output_path, request)
    print(f"Saved requested frame to {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
