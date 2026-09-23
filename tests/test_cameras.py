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
"""Camera layouts, camera-key hints, weight specs and the frozen components."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from conftest import make_policy, tiny_config

from flux_action.models.runtime import _looks_like_local_path, parse_weight_spec, resolve_weights
from flux_action.processing import packing


def test_grid_shape_is_near_square_and_wider_than_tall():
    assert [packing.grid_shape(n) for n in range(1, 7)] == [(1, 1), (1, 2), (2, 2), (2, 2), (2, 3), (2, 3)]


def test_side_by_side_and_grid_canvases():
    cams = torch.rand(2, 1, 3, 48, 64)
    canvas = packing.compose_canvas(cams, "side_by_side", (64, 128))
    assert canvas.shape == (3, 1, 64, 128)
    left = F.interpolate(cams[0], size=(64, 64), mode="bilinear", align_corners=False, antialias=True)
    torch.testing.assert_close(canvas[:, :, :, :64].permute(1, 0, 2, 3), left * 2 - 1)
    three = torch.rand(3, 1, 3, 48, 64)
    grid = packing.compose_canvas(three, "grid", (64, 128))  # 2x2 grid of 32x64 cells, last cell black
    assert grid.shape == (3, 1, 64, 128) and (grid[:, :, 32:, 64:] == -1).all()
    cell = F.interpolate(three[2], size=(32, 64), mode="bilinear", align_corners=False, antialias=True)
    torch.testing.assert_close(grid[:, :, 32:, :64].permute(1, 0, 2, 3), cell * 2 - 1)
    assert packing.content_hw("grid", (64, 128)) == (64, 128) and packing.latent_hw(
        "side_by_side", (64, 128)
    ) == (2, 4)
    with pytest.raises(ValueError, match="side_by_side layout needs two cameras"):
        packing.compose_canvas(three, "side_by_side", (64, 128))
    with pytest.raises(ValueError, match="unknown camera layout"):
        packing.compose_canvas(three, "mosaic", (64, 128))


def test_grid_policy_takes_mixed_resolutions_and_names_missing_cameras():
    policy = make_policy(tiny_config(camera_layout="grid", camera_keys=("images.a", "images.b", "images.c")))
    batch = {
        "images.a": torch.rand(1, 3, 48, 64),
        "images.b": (torch.rand(1, 3, 24, 32) * 255).to(torch.uint8),
        "images.c": torch.rand(1, 3, 40, 40),
    }
    cams = policy._cameras(batch)
    assert cams.shape == (1, 3, 1, 3, 48, 64) and cams.dtype == torch.float32
    with pytest.raises(KeyError, match=r"images\.<camera>"):
        policy._cameras({"images.a": batch["images.a"]})
    single = make_policy()
    with pytest.raises(KeyError, match="wrist, left exterior, right exterior"):
        make_policy(
            tiny_config(
                camera_layout="droid", camera_keys=("images.w", "images.l", "images.r"), canvas_hw=(544, 736)
            )
        )._cameras({})
    assert single.config.camera_layout == "single"


def test_frozen_components_are_unregistered_and_follow_moves():
    policy = make_policy()
    assert policy.frozen.video_vae is policy.video_vae and policy.frozen.text_encoder is policy.text_encoder
    assert not policy.text_encoder.training and not any(
        p.requires_grad for p in policy.text_encoder.parameters()
    )
    assert not any(k.startswith(("video_vae", "text_encoder", "frozen")) for k in policy.state_dict())
    policy.train()
    assert not policy.text_encoder.training  # train() cannot reach the unregistered components
    policy.to(torch.float32)  # device / dtype moves are forwarded without error


def test_weight_specs(tmp_path):
    assert parse_weight_spec("org/repo") == ("org/repo", None, None)
    assert parse_weight_spec("org/repo:temp/dit.safetensors@abc123") == (
        "org/repo",
        "temp/dit.safetensors",
        "abc123",
    )
    assert parse_weight_spec("org/repo@main") == ("org/repo", None, "main")
    with pytest.raises(ValueError, match="weight spec"):
        parse_weight_spec("@abc")
    local = tmp_path / "weights@odd:name.safetensors"
    local.write_bytes(b"")
    assert resolve_weights(str(local), "dit.safetensors") == str(local)  # existing paths are never parsed


def test_missing_local_weights_report_the_path_instead_of_a_hub_repo_id(tmp_path, monkeypatch):
    """The usual first mistake is running a config before downloading its weights."""
    monkeypatch.chdir(tmp_path)  # the relative specs below must not resolve against the repository
    for spec in (
        "outputs/weights/flux-3-action-base.safetensors",  # the path in configs/droid/train.json
        str(tmp_path / "video_vae.safetensors"),
        "./weights",
        "outputs/weights/text_encoder",
    ):
        with pytest.raises(FileNotFoundError, match="no such file or directory"):
            resolve_weights(spec, "dit.safetensors")


def test_hub_specs_are_not_mistaken_for_local_paths():
    """The guard must not intercept a repo_id that simply is not on disk."""
    for spec in ("org/repo", "black-forest-labs/FLUX-3-action", "org/repo:temp/dit.safetensors@abc123"):
        assert not _looks_like_local_path(parse_weight_spec(spec)[0])
