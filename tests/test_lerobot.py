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
"""LeRobot index (v2.1 and v3.0), window streaming, the joint-delta action space with range normalization,
camera dropout and the trainer over an indexed LeRobot dataset."""

import json
import sys

import numpy as np
import pytest
import torch
from conftest import TINY_DIT, FakeVideoVAE
from synthetic_lerobot import CAMERAS, DIM, action_row, build_v3_dataset, build_v21_dataset, state_row

from flux_action import cli
from flux_action.config import PolicyConfig
from flux_action.data.lerobot import index as lerobot_index
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy
from flux_action.processing import normalization
from flux_action.training.checkpoint import export_policy, latest_checkpoint
from flux_action.training.data import WindowDataset, build_dataloader, load_manifest
from flux_action.training.trainer import TrainConfig, Trainer

EPISODES = [
    {"n_frames": 40, "caption": "pick up the cube"},
    {"n_frames": 38, "caption": "put the cube in the bowl"},
    {"n_frames": 44, "caption": "pick up the cube"},
    {"n_frames": 33, "caption": "too short: a window needs frame 1 .. 33"},  # excluded
    {"n_frames": 41, "caption": "stack the cubes"},
]
FRAME_HW = (32, 32)
TINY_POLICY = dict(
    camera_layout="side_by_side",
    camera_keys=["images.top", "images.wrist"],
    canvas_hw=[64, 128],
    action_dim=DIM,
    gripper_flip_dims=[],
    action_parameterization="joint_delta",
    absolute_action_dims=[-1],
    camera_dropout={"images.wrist": 0.2},
    dit_config=TINY_DIT,
    augment=False,
    sampler="euler",
    num_inference_steps=2,
    guidance_scale=1.0,
    sampler_shift=5.0,
)


def tiny_policy(config):
    return FluxActionPolicy(config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))


def _index(root, builder):
    source = builder(root / "source", [dict(e) for e in EPISODES])
    out = root / "index"
    summary = lerobot_index.index_dataset(source, out, CAMERAS)
    return source, out, summary


@pytest.fixture(scope="module")
def v3(tmp_path_factory):
    return _index(tmp_path_factory.mktemp("lerobot_v3"), build_v3_dataset)


@pytest.fixture(scope="module")
def v21(tmp_path_factory):
    return _index(tmp_path_factory.mktemp("lerobot_v21"), build_v21_dataset)


def reference_statistics():
    """q01 / q99 of the joint-delta targets (first row zero, gripper absolute) and of the states."""
    targets, states = [], []
    for e, episode in enumerate(EPISODES):
        if episode["n_frames"] < 34:
            continue
        actions = np.array([action_row(e, i) for i in range(episode["n_frames"])], np.float64)
        delta = np.zeros_like(actions)
        delta[1:] = actions[1:] - actions[:-1]
        delta[:, -1] = actions[:, -1]
        targets.append(delta)
        states.append(np.array([state_row(e, i) for i in range(episode["n_frames"])], np.float64))
    targets, states = np.concatenate(targets), np.concatenate(states)
    return (
        np.percentile(targets, 1, axis=0),
        np.percentile(targets, 99, axis=0),
        np.percentile(states, 1, axis=0),
        np.percentile(states, 99, axis=0),
    )


@pytest.mark.parametrize("layout", ["v3", "v21"])
def test_index_manifest_rows_and_statistics(layout, request):
    source, out, summary = request.getfixturevalue(layout)
    manifest = load_manifest(out / "manifest.json")
    assert manifest["kind"] == "lerobot" and manifest["codebase_version"].startswith(
        "v3" if layout == "v3" else "v2"
    )
    counts = manifest["counts"]
    assert counts["listed"] == 5 and counts["eligible"] == 4 and counts["no_valid_window"] == 1
    assert counts["rows_rejected"] == 0 and summary["rows_rejected"] == {}
    assert [e["episode_index"] for e in manifest["episodes"]] == [0, 1, 2, 4]
    assert counts["valid_starts_total"] == sum(n - 33 for n in (40, 38, 44, 41))
    assert manifest["camera_order"] == ["top", "wrist"] and manifest["camera_hw"] == {
        "top": [48, 64],
        "wrist": [32, 48],
    }
    assert (manifest["state_dim"], manifest["action_dim"]) == (DIM, DIM) and manifest["action_prev"] is True
    assert manifest["action_names"][-1] == "gripper.pos"
    for episode, expected_from in zip(manifest["episodes"], [0, 40, 78, 155], strict=True):
        assert episode["valid_ranges"] == [[1, episode["n_frames"]]]
        assert episode["from_index"] == expected_from
        for camera in ("top", "wrist"):
            assert (source / episode["videos"][camera]["file"]).is_file()
            assert episode["videos"][camera]["first_frame"] == (expected_from if layout == "v3" else 0)
    rows = np.load(out / "rows.f32.npy", mmap_mode="r")
    assert rows.shape == (manifest["total_frames"], 2 * DIM)
    assert rows[78 + 5, :DIM].tolist() == pytest.approx(state_row(2, 5), rel=1e-6)
    assert rows[78 + 5, DIM:].tolist() == pytest.approx(action_row(2, 5), rel=1e-6)
    assert np.isnan(rows[122:155]).all()  # the excluded episode's frames are never read
    statistics = lerobot_index.load_statistics(out / "statistics.json")
    q01, q99, s01, s99 = reference_statistics()
    assert statistics["action"]["parameterization"] == "joint_delta"
    assert statistics["action"]["absolute_dims"] == [DIM - 1] and statistics["clip"] == 6.0
    # the rows are float32; the float64 reference agrees to that precision
    assert statistics["action"]["q01"] == pytest.approx(q01.tolist(), rel=1e-3, abs=1e-4)
    assert statistics["action"]["q99"] == pytest.approx(q99.tolist(), rel=1e-3, abs=1e-4)
    assert statistics["state"]["q01"] == pytest.approx(s01.tolist(), rel=1e-5, abs=1e-4)
    assert statistics["state"]["q99"] == pytest.approx(s99.tolist(), rel=1e-5, abs=1e-4)
    assert statistics["frames"] == 40 + 38 + 44 + 41 and statistics["episodes"] == 4


def test_both_layouts_yield_the_same_rows_and_statistics(v3, v21):
    rows3 = np.load(v3[1] / "rows.f32.npy")
    rows21 = np.load(v21[1] / "rows.f32.npy")
    np.testing.assert_array_equal(np.nan_to_num(rows3), np.nan_to_num(rows21))
    s3, s21 = (json.loads((out / "statistics.json").read_text()) for out in (v3[1], v21[1]))
    assert s3["action"] == s21["action"] and s3["state"] == s21["state"]


@pytest.mark.parametrize("layout", ["v3", "v21"])
def test_windows_carry_previous_command_and_resized_frames(layout, request):
    source, out, _ = request.getfixturevalue(layout)
    manifest = load_manifest(out / "manifest.json")
    dataset = WindowDataset(manifest, source, out / "rows.f32.npy", seed=3, frame_hw=FRAME_HW, decoder="pyav")
    windows = list(dataset)
    assert len(windows) == 4
    for w in windows:
        e, s = w["episode_index"], w["start"]
        assert s >= 1
        assert w["state"].tolist() == pytest.approx(state_row(e, s), rel=1e-6)
        assert w["action"].shape == (32, DIM) and w["action"][0].tolist() == pytest.approx(
            action_row(e, s), rel=1e-6
        )
        assert w["action_prev"].tolist() == pytest.approx(action_row(e, s - 1), rel=1e-6)
        for camera, index in (("top", 0), ("wrist", 1)):
            frames = w[f"images.{camera}"]
            assert frames.dtype == torch.uint8 and frames.shape == (33, 3, *FRAME_HW)  # both cameras rescaled
            # flat colors survive AV1 and the rescale: red = episode, blue = camera
            assert abs(int(frames[:, 0].float().mean()) - (30 + 50 * e)) <= 6
            assert abs(int(frames[:, 2].float().mean()) - (60 + 60 * index)) <= 6
    batch = next(iter(build_dataloader(dataset, in_process=True)))
    assert batch["action_prev"].shape == (1, DIM) and batch["images.wrist"].shape == (1, 33, 3, *FRAME_HW)


def test_policy_normalizes_deltas_and_integrates_back(v3):
    source, out, _ = v3
    manifest = load_manifest(out / "manifest.json")
    statistics = lerobot_index.load_statistics(out / "statistics.json")
    bounds = {k: {b: statistics[k][b] for b in ("q01", "q99")} for k in ("action", "state")}
    config = PolicyConfig(
        **{**TINY_POLICY, "action_normalization": bounds["action"], "state_normalization": bounds["state"]}
    )
    policy = tiny_policy(config).train()
    dataset = WindowDataset(
        manifest, source, out / "rows.f32.npy", seed=1, frame_hw=FRAME_HW, decoder="pyav", windows_per_rank=2
    )
    batch = next(iter(build_dataloader(dataset, in_process=True)))
    prepared = policy.prepare(batch)
    # reference: deltas from the previous command, gripper absolute, then range normalization with clip 6
    q01, q99 = (np.array(bounds["action"][k]) for k in ("q01", "q99"))
    span = np.where(q99 - q01 > 1e-6, q99 - q01, 1.0)
    actions, previous = (
        batch["action"].numpy().astype(np.float64),
        batch["action_prev"].numpy().astype(np.float64),
    )
    shifted = np.concatenate([previous[:, None], actions[:, :-1]], axis=1)
    targets = actions - shifted
    targets[..., -1] = actions[..., -1]
    expected = np.clip(2 * (targets - q01) / span - 1, -6, 6)
    torch.testing.assert_close(prepared.actions.double(), torch.from_numpy(expected), rtol=1e-5, atol=1e-5)
    s01, s99 = (np.array(bounds["state"][k]) for k in ("q01", "q99"))
    s_span = np.where(s99 - s01 > 1e-6, s99 - s01, 1.0)
    expected_state = np.clip(2 * (batch["state"].numpy().astype(np.float64) - s01) / s_span - 1, -6, 6)
    torch.testing.assert_close(
        prepared.state.double(), torch.from_numpy(expected_state), rtol=1e-5, atol=1e-5
    )
    loss, info = policy(batch, prepared=prepared)
    assert torch.isfinite(loss) and info["n_valid_windows"] == 2
    # inference undoes the normalization and integrates the deltas onto the observed state
    observation = {
        k: (v[:, 0] if k.startswith("images.") else v) for k, v in batch.items() if k != "action_prev"
    }
    policy.eval()
    known = torch.linspace(-0.5, 0.5, 32 * DIM).reshape(32, DIM)
    policy._sample = lambda cond, caption, seed: known.clone()  # noqa: E731  (the DiT is not under test)
    predicted = policy.predict_action_chunk(observation)
    denormalized = (known.double() + 1) * torch.from_numpy(span) / 2 + torch.from_numpy(q01)
    for i in range(2):
        state = batch["state"][i].double()
        expected_abs = state[None] + torch.cumsum(denormalized, dim=0)
        expected_abs[:, -1] = denormalized[:, -1]
        torch.testing.assert_close(predicted[i].double(), expected_abs, rtol=1e-5, atol=1e-4)
    assert predicted.shape == (2, 32, DIM)
    with pytest.raises(AssertionError, match="action_prev"):
        policy.train().prepare({k: v for k, v in batch.items() if k != "action_prev"})


def test_camera_dropout_is_deterministic_and_gray():
    policy = tiny_policy(PolicyConfig(**{**TINY_POLICY, "camera_dropout": {"images.wrist": 1.0}}))
    cams = torch.randint(0, 255, (2, 3, 3, 8, 8), dtype=torch.uint8)
    dropped = FluxActionPolicy._drop_cameras(cams, [1])
    assert torch.equal(dropped[0], cams[0]) and (dropped[1] == 128).all()
    assert FluxActionPolicy._drop_cameras(cams, []) is cams
    floats = FluxActionPolicy._drop_cameras(cams.float() / 255, [0])
    assert torch.allclose(floats[0], torch.full_like(floats[0], 128 / 255))
    # the draw comes from the window seed: same seed, same decision; the gray tile reaches the encoder
    seen = []
    policy._encode_windows = lambda cams, idx, device, generators, dropped=None: (
        seen.append(dropped),
        torch.zeros(len(idx), 96, 9, 2, 4),
    )[1]  # noqa: E731
    rng = torch.Generator().manual_seed(0)
    batch = {
        "images.top": torch.randint(0, 255, (3, 33, 3, 8, 8), dtype=torch.uint8, generator=rng),
        "images.wrist": torch.randint(0, 255, (3, 33, 3, 8, 8), dtype=torch.uint8, generator=rng),
        "state": torch.zeros(3, DIM),
        "action": torch.zeros(3, 32, DIM),
        "action_prev": torch.zeros(3, DIM),
        "task": ["a", "b", "c"],
        "window_seed": torch.tensor([11, 12, 13]),
    }
    policy.train().prepare(batch)
    assert seen[-1] == [[1], [1], [1]]
    half = tiny_policy(PolicyConfig(**{**TINY_POLICY, "camera_dropout": {"images.wrist": 0.5}}))
    half._encode_windows = policy._encode_windows
    half.train().prepare(batch)
    first = seen[-1]
    half.prepare(batch)
    assert seen[-1] == first and any(d == [] for d in first) or seen[-1] == first
    half.eval()
    half.prepare(batch)
    assert seen[-1] == [[], [], []]  # no dropout outside training


def make_config(source, out, output, **overrides):
    values = dict(
        source_root=str(source),
        index_dir=str(out),
        output_dir=str(output),
        policy=TINY_POLICY,
        steps=4,
        cooldown_start=None,
        cooldown_steps=0,
        windows_per_rank=2,
        num_workers=1,
        in_process_loader=True,
        decoder="pyav",
        frame_hw=FRAME_HW,
        param_dtype="float32",
        compute_dtype="float32",
        activation_checkpointing=False,
        checkpoint_every=2,
        log_every=1,
        max_grad_norm=1e-3,
        ema_sigma_rels=(0.10,),
        reseed_on_resume=False,  # the resume test compares an interrupted run with an uninterrupted one
    )
    values.update(overrides)
    return TrainConfig(**values)


def records(output):
    return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]


def test_trainer_runs_resumes_and_exports_with_the_index_bounds(v3, tmp_path):
    source, out, _ = v3
    output = tmp_path / "run"
    result = Trainer(make_config(source, out, output), policy_factory=tiny_policy).run()
    assert result["step"] == 4 and latest_checkpoint(output) == output / "step-4"
    lines = records(output)
    event = next(r for r in lines if r.get("event") == "normalization")
    assert set(event["adopted"]) == {"action_normalization", "state_normalization"}
    assert event["action_parameterization"] == "joint_delta" and event["absolute_action_dims"] == [DIM - 1]
    steps = [r for r in lines if "loss" in r]
    assert [r["step"] for r in steps] == [1, 2, 3, 4] and all(np.isfinite(r["loss"]) for r in steps)
    assert all(0 < r["grad_clip"] < 1 and r["grad_norm"] > 1e-3 for r in steps)  # clipped at 1e-3
    statistics = lerobot_index.load_statistics(out / "statistics.json")
    saved = json.loads((output / "step-4" / "config.json").read_text())
    assert saved["action_normalization"]["q99"] == pytest.approx(statistics["action"]["q99"])
    assert saved["state_normalization"]["q01"] == pytest.approx(statistics["state"]["q01"])
    assert saved["action_parameterization"] == "joint_delta"
    assert lerobot_index.normalize_dims(saved["absolute_action_dims"], DIM) == (DIM - 1,)
    # exact resume reproduces the uninterrupted losses
    part = tmp_path / "part"
    Trainer(make_config(source, out, part, steps=2), policy_factory=tiny_policy).run()
    Trainer(make_config(source, out, part, steps=4), policy_factory=tiny_policy).run()
    assert [(r["step"], r["loss"]) for r in records(part) if "loss" in r] == [
        (r["step"], r["loss"]) for r in steps
    ]
    # export carries the bounds and predicts absolute commands from a dataset window
    export = export_policy(output / "step-4", tmp_path / "export", profile="ema_0p10")
    assert export["profile"] == "ema_0p10"
    restored = FluxActionPolicy.from_pretrained(
        tmp_path / "export", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    assert restored.config.action_normalization["q01"] == pytest.approx(statistics["action"]["q01"])
    manifest = load_manifest(out / "manifest.json")
    dataset = WindowDataset(manifest, source, out / "rows.f32.npy", seed=9, frame_hw=FRAME_HW, decoder="pyav")
    batch = next(iter(build_dataloader(dataset, in_process=True)))
    observation = {
        k: (v[:, 0] if k.startswith("images.") else v) for k, v in batch.items() if k != "action_prev"
    }
    chunk = restored.predict_action_chunk(observation)
    assert chunk.shape == (1, 32, DIM) and torch.isfinite(chunk).all()
    assert torch.equal(restored.select_action(observation), chunk[0, 0][None])


def test_trainer_refuses_a_policy_that_disagrees_with_the_index(v3, tmp_path):
    source, out, _ = v3
    statistics = lerobot_index.load_statistics(out / "statistics.json")
    other = {"q01": [0.0] * DIM, "q99": [1.0] * DIM}
    with pytest.raises(ValueError, match="disagrees"):
        Trainer(
            make_config(source, out, tmp_path / "a", policy={**TINY_POLICY, "action_normalization": other}),
            policy_factory=tiny_policy,
        ).run()
    with pytest.raises(ValueError, match="joint_delta"):
        Trainer(
            make_config(
                source,
                out,
                tmp_path / "b",
                policy={**TINY_POLICY, "action_parameterization": "absolute", "absolute_action_dims": []},
            ),
            policy_factory=tiny_policy,
        ).run()
    with pytest.raises(ValueError, match="absolute dims"):
        Trainer(
            make_config(source, out, tmp_path / "c", policy={**TINY_POLICY, "absolute_action_dims": []}),
            policy_factory=tiny_policy,
        ).run()
    with pytest.raises(ValueError, match="action_dim"):
        Trainer(
            make_config(source, out, tmp_path / "d", policy={**TINY_POLICY, "action_dim": 8}),
            policy_factory=tiny_policy,
        ).run()
    # the same bounds spelled out explicitly are accepted, and fields left out are taken from the index
    explicit = {
        **TINY_POLICY,
        "action_normalization": {k: statistics["action"][k] for k in ("q01", "q99")},
        "state_normalization": {k: statistics["state"][k] for k in ("q01", "q99")},
    }
    del explicit["absolute_action_dims"]
    Trainer(
        make_config(source, out, tmp_path / "e", policy=explicit, steps=1), policy_factory=tiny_policy
    ).run()


def test_policy_normalization_math():
    cfg = PolicyConfig(
        action_dim=6, gripper_flip_dims=(), action_normalization={"q01": [0] * 6, "q99": [2] * 6}
    )
    assert cfg.action_normalization == {"q01": [0.0] * 6, "q99": [2.0] * 6}
    x = torch.tensor([[0.0, 1.0, 2.0, 4.0, -4.0, 1.0]])
    normalized = normalization.normalize(x, cfg.action_normalization, cfg.normalization_clip)
    assert normalized.tolist() == [[-1.0, 0.0, 1.0, 3.0, -5.0, 0.0]]
    torch.testing.assert_close(normalization.denormalize(normalized, cfg.action_normalization), x)
    degenerate = {"q01": [1.0] * 6, "q99": [1.0] * 6}  # a channel that never moved keeps span 1
    assert normalization.normalize(torch.full((1, 6), 1.5), degenerate, 6.0).tolist() == [[0.0] * 6]


def test_index_lerobot_command(v3, tmp_path, monkeypatch, capsys):
    source = v3[0]
    out = tmp_path / "cli_index"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flux-action",
            "index-lerobot",
            "--source-root",
            str(source),
            "--output-dir",
            str(out),
            "--camera",
            "top=observation.images.top",
            "--camera",
            "wrist=observation.images.front",
            "--dataset-id",
            "lab/synthetic",
        ],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["eligible"] == 4 and result["action"]["parameterization"] == "joint_delta"
    manifest = load_manifest(out / "manifest.json")
    assert manifest["dataset_id"] == "lab/synthetic" and (out / "statistics.json").is_file()
    assert (out / "rows.f32.npy").is_file()


MOTION_EPISODES = [
    {"n_frames": 120, "caption": "moving then still", "stride": 1.0, "freeze_from": 30},
    {"n_frames": 120, "caption": "never moves", "stride": 0.0},
    {"n_frames": 200, "caption": "moving throughout", "stride": 1.0},
    {"n_frames": 80, "caption": "held out", "stride": 1.0},
]


@pytest.fixture(scope="module")
def curated(tmp_path_factory):
    root = tmp_path_factory.mktemp("lerobot_curated")
    source = build_v3_dataset(root / "source", [dict(e) for e in MOTION_EPISODES])
    out = root / "index"
    summary = lerobot_index.index_dataset(source, out, CAMERAS, val_episodes=1, strip_dead_windows=0.6)
    return source, out, summary


def test_dead_window_filter_and_validation_split(curated):
    source, out, summary = curated
    manifest = load_manifest(out / "manifest.json")
    counts = manifest["counts"]
    assert counts["dead_static"] == 1 and counts["dead_long"] == 0 and counts["dead_no_window"] == 0
    assert [e["episode_index"] for e in manifest["episodes"]] == [0, 2, 3]
    still, moving, held = manifest["episodes"]
    # the state freezes at frame 30 (frame 30 itself still moved into place), so frames 31.. are dead; a
    # 33-frame window from s holds s + 2 dead frames -> s <= 17 keeps at most 60% dead
    assert still["valid_ranges"] == [[1, 50]] and still["n_valid_starts"] == 17
    assert moving["valid_ranges"] == [[1, 200]] and moving["n_valid_starts"] == 200 - 33
    assert counts["dead_starts_removed"] == (120 - 33) - 17
    assert (still["split"], moving["split"], held["split"]) == ("train", "train", "val")
    assert counts["train_episodes"] == 2 and counts["val_episodes"] == 1
    assert manifest["dead_window_filter"]["max_dead_fraction"] == 0.6
    train = WindowDataset(manifest, source, out / "rows.f32.npy", seed=1, frame_hw=FRAME_HW, decoder="pyav")
    assert [e["episode_index"] for e in train.episodes] == [0, 2]
    assert train.valid_starts(0) == list(range(1, 18))
    for w in train:
        if w["episode_index"] == 0:
            assert 1 <= w["start"] <= 17
    val = WindowDataset(
        manifest, source, out / "rows.f32.npy", seed=1, frame_hw=FRAME_HW, decoder="pyav", split="val"
    )
    assert [e["episode_index"] for e in val.episodes] == [3]
    everything = WindowDataset(
        manifest, source, out / "rows.f32.npy", seed=1, frame_hw=FRAME_HW, decoder="pyav", split=None
    )
    assert len(everything.episodes) == 3
    # statistics: over every kept episode (train and val), as the corpus builder computed them
    statistics = lerobot_index.load_statistics(out / "statistics.json")
    assert statistics["episodes"] == 3 and statistics["frames"] == 120 + 200 + 80
    assert statistics["action"]["q99"][0] == pytest.approx(1.01, abs=1e-3)  # stride 1 + gripper/100 coupling


def test_dead_mask_definition():
    fps = 30
    states = np.zeros((300, DIM))
    states[:100, 0] = np.arange(100) * 1.0  # moving for 100 frames, then holding still for 200 (>= 2 s)
    states[100:, 0] = 99.0
    mask, moving = lerobot_index.dead_mask(states, fps)
    assert moving == pytest.approx(100 / 300) and not mask[:100].any() and mask[100:].all()
    short = np.zeros((100, DIM))
    short[::2, 0] = 1.0  # alternating: never still for 2 s
    mask, _ = lerobot_index.dead_mask(short, fps)
    assert not mask.any()
    assert lerobot_index._ranges_of_starts([1, 2, 3, 7, 8], 32) == [[1, 36], [7, 41]]


def test_visits_per_epoch_adds_windows_deterministically(v3):
    source, out, _ = v3
    manifest = load_manifest(out / "manifest.json")
    common = dict(seed=5, frame_hw=FRAME_HW, decoder="pyav")
    once = list(WindowDataset(manifest, source, out / "rows.f32.npy", **common))
    thrice = WindowDataset(manifest, source, out / "rows.f32.npy", visits_per_epoch=3, **common)
    assert thrice.n_positions == 12 and thrice.batches_per_rank == 12
    windows = list(thrice)
    key = lambda w: (w["episode_index"], w["start"], w["window_seed"])  # noqa: E731
    assert len(windows) == 12 and {key(w) for w in once} <= {key(w) for w in windows}
    assert len({key(w) for w in windows}) == 12  # every visit draws its own start and seed
    again = list(WindowDataset(manifest, source, out / "rows.f32.npy", visits_per_epoch=3, **common))
    assert [key(w) for w in again] == [key(w) for w in windows]
    with pytest.raises(ValueError, match="visits_per_epoch"):
        WindowDataset(manifest, source, out / "rows.f32.npy", windows_per_rank=8, **common)
    WindowDataset(manifest, source, out / "rows.f32.npy", windows_per_rank=8, visits_per_epoch=2, **common)
    with pytest.raises(ValueError, match="visits_per_epoch"):
        TrainConfig(source_root="s", index_dir="i", output_dir="o", visits_per_epoch=0)


def test_gray_camera_stream(tmp_path):
    source = build_v3_dataset(tmp_path / "source", [dict(e) for e in EPISODES[:2]])
    out = tmp_path / "index"
    lerobot_index.index_dataset(source, out, {"top": CAMERAS["top"], "wrist": lerobot_index.GRAY})
    manifest = load_manifest(out / "manifest.json")
    assert manifest["cameras"] == {"top": CAMERAS["top"], "wrist": "gray"}
    assert manifest["camera_hw"] == {"top": [48, 64], "wrist": None}
    assert all(e["videos"]["wrist"] is None for e in manifest["episodes"])
    dataset = WindowDataset(manifest, source, out / "rows.f32.npy", seed=1, frame_hw=FRAME_HW, decoder="pyav")
    window = next(iter(dataset))
    assert window["images.wrist"].shape == (33, 3, *FRAME_HW) and (window["images.wrist"] == 128).all()
    assert window["images.top"].shape == (33, 3, *FRAME_HW) and not (window["images.top"] == 128).all()
    policy = tiny_policy(PolicyConfig(**TINY_POLICY)).train()
    batch = next(iter(build_dataloader(dataset, in_process=True)))
    loss, info = policy(batch)
    assert torch.isfinite(loss) and info["n_valid_windows"] == 1
    with pytest.raises(ValueError, match="at least one camera must be a video"):
        lerobot_index.build_manifest(source, {"top": lerobot_index.GRAY})


def test_evaluate_export_scores_held_out_windows(curated, tmp_path, monkeypatch, capsys):
    from flux_action.inference.evaluate import evaluate_policy, spread

    assert (
        spread(10, 4) == [0, 3, 6, 9] and spread(1, 4) == [0] and spread(0, 3) == [] and spread(5, 1) == [0]
    )
    source, out, _ = curated
    statistics = lerobot_index.load_statistics(out / "statistics.json")
    config = PolicyConfig(
        **{
            **TINY_POLICY,
            "action_normalization": {k: statistics["action"][k] for k in ("q01", "q99")},
            "state_normalization": {k: statistics["state"][k] for k in ("q01", "q99")},
        }
    )
    policy = tiny_policy(config).eval()
    policy.save_pretrained(tmp_path / "export")
    report = evaluate_policy(policy, source, out, windows_per_episode=3, frame_hw=FRAME_HW)
    assert report["split"] == "val" and report["episodes"] == 1 and report["n_windows"] == 3
    assert [w["start"] for w in report["windows"]] == [1, 24, 47]  # spread over the 47 allowed starts
    assert np.isfinite(report["action_mse_normalized"]) and np.isfinite(report["action_mse_raw"])
    assert len(report["action_mse_raw_per_channel"]) == DIM
    # the recorded targets integrate back onto the command before the window exactly; onto the observed
    # state (the deployment convention) they carry the state-vs-command offset of the first command
    manifest = load_manifest(out / "manifest.json")
    dataset = WindowDataset(
        manifest, source, out / "rows.f32.npy", seed=0, frame_hw=FRAME_HW, decoder="pyav", split="val"
    )
    batch = next(iter(build_dataloader(dataset, in_process=True)))
    recorded = policy.training_targets(batch)
    torch.testing.assert_close(
        policy.actions_from_targets(recorded, batch["action_prev"]),
        batch["action"].float(),
        rtol=1e-4,
        atol=1e-3,
    )
    onto_state = policy.actions_from_targets(recorded, batch["state"])
    offset = (batch["action_prev"] - batch["state"])[:, None, :-1]
    torch.testing.assert_close(
        onto_state[..., :-1], batch["action"].float()[..., :-1] - offset, rtol=1e-4, atol=1e-3
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flux-action",
            "evaluate",
            "--checkpoint",
            str(tmp_path),
            "--subfolder",
            "export",
            "--source-root",
            str(source),
            "--index-dir",
            str(out),
            "--device",
            "cpu",
            "--frame-hw",
            "32,32",
            "--windows-per-episode",
            "2",
            "--setting",
            "num_inference_steps=1",
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    restore = FluxActionPolicy.from_pretrained  # the module attribute is this very class: keep the original
    monkeypatch.setattr(
        "flux_action.inference.evaluate.FluxActionPolicy.from_pretrained",
        lambda path, video_vae=None, text_encoder=None, device="cpu": restore(
            path, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32), device=device
        ),
    )
    cli.main()
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / "report.json").read_text())
    assert (
        printed["n_windows"] == 2
        and saved["inference"]["num_inference_steps"] == 1
        and len(saved["windows"]) == 2
    )
    assert saved["checkpoint_manifest"]["kind"] == "policy_export"


def test_trainer_requires_statistics_before_model_construction(v3, tmp_path):
    source, index, _ = v3
    copied = tmp_path / "index"
    copied.mkdir()
    for name in ("manifest.json", "rows.f32.npy"):
        (copied / name).write_bytes((index / name).read_bytes())
    with pytest.raises(FileNotFoundError, match="normalization statistics"):
        Trainer(make_config(source, copied, tmp_path / "run"), policy_factory=tiny_policy).build()


@pytest.mark.parametrize("change", ["statistics", "cameras", "dataset"])
def test_resume_rejects_changed_contract_before_restoring_weights(v3, tmp_path, monkeypatch, change):
    source, index, _ = v3
    copied = tmp_path / "index"
    copied.mkdir()
    for name in ("manifest.json", "rows.f32.npy", "statistics.json"):
        (copied / name).write_bytes((index / name).read_bytes())
    output = tmp_path / "run"
    Trainer(make_config(source, copied, output, steps=2), policy_factory=tiny_policy).run()
    config = make_config(source, copied, output)
    if change == "statistics":
        path = copied / "statistics.json"
        data = json.loads(path.read_text())
        data["action"]["q99"][0] += 10
        path.write_text(json.dumps(data))
    elif change == "cameras":
        config.policy = {**TINY_POLICY, "camera_keys": list(reversed(TINY_POLICY["camera_keys"]))}
    else:
        path = copied / "manifest.json"
        data = json.loads(path.read_text())
        data["dataset_revision"] = "different-revision"
        path.write_text(json.dumps(data))

    def unexpected_restore(*args, **kwargs):
        pytest.fail("contract must be checked before model/optimizer restoration")

    monkeypatch.setattr("flux_action.training.trainer.load_checkpoint", unexpected_restore)
    with pytest.raises(ValueError, match="policy/data contract"):
        Trainer(config, policy_factory=tiny_policy).build()


@pytest.mark.parametrize("layout", ["v3", "v21"])
def test_history_windows_align_states_preceding_commands_and_episode_boundaries(layout, request):
    source, out, _ = request.getfixturevalue(layout)
    dataset = WindowDataset(
        load_manifest(out / "manifest.json"),
        source,
        out / "rows.f32.npy",
        seed=3,
        frame_hw=FRAME_HW,
        decoder="pyav",
        n_obs_steps=8,
    )
    assert {e["episode_index"] for e in dataset.episodes} == {2, 4}  # shorter episodes cannot supply history
    for position, episode in enumerate(dataset.episodes):
        assert dataset.valid_starts(position)[0] == 8
        with pytest.raises(ValueError, match="allowed window start"):
            dataset.window_at(position, 7)
        for start in dataset.valid_starts(position):
            item = dataset.window_at(position, start)
            e = episode["episode_index"]
            torch.testing.assert_close(
                item["state"],
                torch.tensor([state_row(e, f) for f in range(start - 7, start + 1)], dtype=torch.float32),
            )
            torch.testing.assert_close(
                item["command_history"],
                torch.tensor([action_row(e, f) for f in range(start - 8, start)], dtype=torch.float32),
            )
            torch.testing.assert_close(item["action_prev"], item["command_history"][-1])
            assert item["action"][0].tolist() == pytest.approx(action_row(e, start), rel=1e-6)
            assert item["images.top"].shape == (40, 3, *FRAME_HW)


def test_history_trainer_resume_and_offline_evaluation(v3, tmp_path, monkeypatch):
    from flux_action.inference.evaluate import evaluate_policy

    source, out, _ = v3
    fields = {
        **TINY_POLICY,
        "inference_profile": "history",
        "n_obs_steps": 8,
        "history_snapshots": 2,
        "condition_on_past_actions": True,
        "separate_timesteps": True,
        "conditioning_noise_max": 0.2,
        "loss_reduction": "modalities",
        "action_loss_weight": 0.5,
        "action_channel_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
    }
    full = Trainer(make_config(source, out, tmp_path / "full", policy=fields), policy_factory=tiny_policy)
    full.run()
    Trainer(
        make_config(source, out, tmp_path / "split", policy=fields, steps=2), policy_factory=tiny_policy
    ).run()
    resumed = Trainer(make_config(source, out, tmp_path / "split", policy=fields), policy_factory=tiny_policy)
    resumed.run()
    assert resumed.state.extra["resumes"][-1]["mode"] == "exact"
    for key, value in full.policy.state_dict().items():
        torch.testing.assert_close(resumed.policy.state_dict()[key], value, rtol=0, atol=0)
    for key, ema in full.emas.items():
        for actual, expected in zip(resumed.emas[key].shadow, ema.shadow, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The export rebuilds the DiT from the saved config: the 12-channel history conditioning head
    # (state + previous command) must load, and the restored policy must keep the history contract.
    export = export_policy(
        latest_checkpoint(tmp_path / "full"), tmp_path / "history-export", profile="ema_0p10"
    )
    assert export["profile"] == "ema_0p10"
    restored = FluxActionPolicy.from_pretrained(
        tmp_path / "history-export", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    assert (restored.config.n_obs_steps, restored.config.condition_on_past_actions) == (8, True)
    assert restored.dit.state_dict()["emb_in.action_cond.weight"].shape[1] == 2 * DIM
    # A perfect target predictor must reconstruct recorded commands from the last history command,
    # not from the measured follower state. This checks the offline evaluation input/anchor too.
    policy = full.policy.eval()
    dataset = WindowDataset(
        load_manifest(out / "manifest.json"),
        source,
        out / "rows.f32.npy",
        seed=0,
        frame_hw=FRAME_HW,
        decoder="pyav",
        n_obs_steps=8,
    )
    from flux_action.training.data import collate_windows

    samples = [
        collate_windows([dataset.window_at(i, dataset.valid_starts(i)[0])])
        for i in range(len(dataset.episodes))
    ]
    targets = iter(policy.training_targets(sample) for sample in samples)

    def predict(observation):
        assert observation["state"].shape == (1, 8, 6)
        assert observation["images.top"].shape[1] == 8
        assert observation["command_history"].shape == (1, 8, 6)
        return next(targets).to(observation["state"].device)  # a real predictor answers on the policy device

    monkeypatch.setattr(policy, "predict_normalized_targets", predict)
    report = evaluate_policy(policy, source, out, split="train", windows_per_episode=1, frame_hw=FRAME_HW)
    assert report["action_mse_normalized"] == 0.0
    assert report["action_mse_raw"] < 1e-6
    with pytest.raises(ValueError, match="policy/data contract"):
        Trainer(
            make_config(source, out, tmp_path / "split", policy={**fields, "conditioning_noise_max": 0.1}),
            policy_factory=tiny_policy,
        ).build()


def test_history_recipe_requires_matching_index_horizon(v3, tmp_path):
    source, out, _ = v3

    def unexpected(config):
        pytest.fail("index mismatch must fail before model construction")

    with pytest.raises(ValueError, match="re-index with --chunk-size 42"):
        Trainer(
            make_config(source, out, tmp_path / "run", policy={**TINY_POLICY, "chunk_size": 42}),
            policy_factory=unexpected,
        ).build()
