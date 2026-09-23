# Copyright 2026 Black Forest Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Frame-exact window decoding from the compact Cosmos3-DROID AV1 files.

The files hold several episodes back to back with a keyframe every second frame, so a window is
read by seeking to the keyframe at or before its first frame and decoding forward. Two backends:

* ``ffmpeg`` (default): the ``ffmpeg`` binary with an accurate input seek half a frame before the
  target. Its RGB conversion is the one ``prepare-droid`` used, so windows equal the prepared
  episodes byte for byte on the same machine.
* ``pyav``: in-process through PyAV's bundled FFmpeg. About three times faster and free of
  subprocesses, but its libswscale version may differ from the system binary's, which moves RGB
  values by up to a few levels. Identical frames, not identical bytes, across backends.
"""

import shutil
import subprocess
from fractions import Fraction

import numpy as np

DECODERS = ("ffmpeg", "pyav")
FRAME_HW = (360, 640)


def decode_window(
    path,
    first_frame: int,
    length: int,
    *,
    fps: int = 15,
    frame_hw: tuple[int, int] = FRAME_HW,
    decoder: str = "ffmpeg",
    threads: int = 1,
    resize: bool = False,
) -> np.ndarray:
    """``length`` RGB frames from file frame ``first_frame`` on: uint8 ``(length, H, W, 3)``.

    ``frame_hw`` is the size the frames must have. With ``resize`` the decoded frames are scaled to it
    (libswscale bilinear in both backends, the filter the SO-101 corpus was built with); without it, frames
    of another size are an error, so a DROID window is never silently rescaled.
    """
    if first_frame < 0 or length < 1:
        raise ValueError("first_frame must be non-negative and length positive")
    if decoder == "ffmpeg":
        return _decode_ffmpeg(str(path), first_frame, length, fps, frame_hw, threads, resize)
    if decoder == "pyav":
        return _decode_pyav(str(path), first_frame, length, fps, frame_hw, threads, resize)
    raise ValueError(f"unknown decoder {decoder!r}; choose from {DECODERS}")


def _decode_ffmpeg(path, first_frame, length, fps, frame_hw, threads, resize):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("the ffmpeg binary is not on PATH; install it or use decoder='pyav'")
    h, w = frame_hw
    # Accurate seeking keeps the first frame whose timestamp is >= -ss. Half a frame early lands
    # exactly on ``first_frame`` without depending on decimal rounding of first_frame / fps.
    seek = max(0.0, (first_frame - 0.5) / fps)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        str(threads),
        "-filter_threads",
        str(threads),
        "-ss",
        f"{seek:.6f}",
        "-i",
        path,
        "-frames:v",
        str(length),
        "-fps_mode",
        "passthrough",
        *(["-vf", f"scale={w}:{h}:flags=bilinear"] if resize else []),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    expected = length * h * w * 3
    if len(result.stdout) != expected:
        raise ValueError(
            f"{path}: expected {length} frames of {h}x{w} from frame {first_frame}, "
            f"got {len(result.stdout) // (h * w * 3)}"
        )
    return np.frombuffer(bytearray(result.stdout), np.uint8).reshape(length, h, w, 3)  # writable buffer


def _decode_pyav(path, first_frame, length, fps, frame_hw, threads, resize):
    import av

    h, w = frame_hw
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = threads
        time_base = stream.time_base
        container.seek(
            int(Fraction(first_frame, fps) / time_base), stream=stream, backward=True, any_frame=False
        )
        for frame in container.decode(stream):
            index = round(float(frame.pts * time_base) * fps)
            if index < first_frame:
                continue
            if index != first_frame + len(frames):
                raise ValueError(
                    f"{path}: frame {first_frame + len(frames)} missing, decoder produced {index}"
                )
            if resize:
                frames.append(frame.to_ndarray(format="rgb24", width=w, height=h))
            else:
                if (frame.height, frame.width) != (h, w):
                    raise ValueError(f"{path}: frames are {frame.height}x{frame.width}, expected {frame_hw}")
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == length:
                break
    if len(frames) != length:
        raise ValueError(f"{path}: only {len(frames)} of {length} frames from frame {first_frame}")
    return np.stack(frames)
