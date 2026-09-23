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
"""Seek-decoded windows equal sequential decodes for both backends and land on the right frames."""

import shutil

import numpy as np
import pytest
from synthetic_droid import FRAME_HW, frame_image, write_av1_clip

from flux_action.data.droid.cosmos import decode_frames
from flux_action.data.droid.video import decode_window


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    frames = np.stack([frame_image(1, i, 2) for i in range(45)])
    write_av1_clip(path, frames)
    return path


def sequential_pyav(path):
    import av

    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


@pytest.mark.parametrize("decoder", ["ffmpeg", "pyav"])
def test_seek_decode_matches_sequential_decode(clip, decoder):
    if decoder == "ffmpeg" and shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary not installed")
    if decoder == "ffmpeg":
        reference = decode_frames(clip, 0, 45, FRAME_HW)
    else:
        reference = sequential_pyav(clip)
    assert reference.shape == (45, *FRAME_HW, 3)
    for first in (0, 1, 7, 12):
        window = decode_window(clip, first, 33, frame_hw=FRAME_HW, decoder=decoder)
        assert window.shape == (33, *FRAME_HW, 3)
        assert np.array_equal(window, reference[first : first + 33]), (decoder, first)
        # green encodes the frame index; a wrong seek shows up as a wrong ramp
        greens = window[..., 1].reshape(33, -1).mean(1)
        assert np.abs(greens - (6 * np.arange(first, first + 33)) % 250).max() < 4
    with pytest.raises(ValueError):
        decode_window(clip, 20, 33, frame_hw=FRAME_HW, decoder=decoder)
    with pytest.raises(ValueError):
        decode_window(clip, -1, 3, frame_hw=FRAME_HW, decoder=decoder)


def test_unknown_decoder(clip):
    with pytest.raises(ValueError, match="unknown decoder"):
        decode_window(clip, 0, 1, decoder="magic")
