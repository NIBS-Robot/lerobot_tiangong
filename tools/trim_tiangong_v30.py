#!/usr/bin/env python

"""Trim static starts and final gripper-retreat tails from a LeRobot v3 dataset.

The source dataset is never modified. Every episode keeps its episode index and
data/video file mapping, while data rows and MP4 frames are trimmed to the same
``[start_frame, end_frame)`` window.

Example:

    PYTHONPATH=src python tools/trim_tiangong_v30.py \
        --source data/tiangong_merged_v30 \
        --output data/tiangong_merged_v30_trimmed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.io_utils import write_info, write_table_one_row_group_per_episode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DatasetInfo


GRIPPER_ACTION_COLUMNS = (9, 19)


@dataclass(frozen=True)
class TrimBounds:
    start: int
    end: int
    source_length: int
    has_release: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/tiangong_merged_v30"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-tolerance", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing output directory and skip valid encoded videos.",
    )
    return parser.parse_args()


def _trim_bounds(action: np.ndarray, state: np.ndarray, tolerance: float) -> TrimBounds:
    if action.ndim != 2 or action.shape[1] < 20:
        raise ValueError(f"Expected action shape (N, >=20), got {action.shape}")
    if state.ndim != 2 or state.shape[1] < 34:
        raise ValueError(f"Expected observation.state shape (N, >=34), got {state.shape}")
    if len(action) != len(state) or len(action) < 2:
        raise ValueError(f"Action/state lengths are invalid: {len(action)} vs {len(state)}")

    pose_error = np.maximum(
        np.abs(action[:, :9] - state[:, 14:23]).max(axis=1),
        np.abs(action[:, 10:19] - state[:, 23:32]).max(axis=1),
    )
    gripper_commands = action[:, GRIPPER_ACTION_COLUMNS] > 0.5
    gripper_changes = np.r_[
        np.zeros((1, 2), dtype=bool),
        gripper_commands[1:] != gripper_commands[:-1],
    ]
    moving = (pose_error > tolerance) | gripper_changes.any(axis=1)
    start = int(np.flatnonzero(moving)[0]) if moving.any() else len(action)

    release_frames = []
    for column in GRIPPER_ACTION_COLUMNS:
        closed = action[:, column] > 0.5
        releases = np.flatnonzero(closed[:-1] & ~closed[1:]) + 1
        if len(releases):
            release_frames.append(int(releases[-1]))

    has_release = bool(release_frames)
    end = max(release_frames) if has_release else len(action)
    if start >= end:
        raise ValueError(
            f"Empty trim window for episode with {len(action)} frames: start={start}, end={end}"
        )
    return TrimBounds(start=start, end=end, source_length=len(action), has_release=has_release)


def _data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")
    return files


def _load_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata found under {root / 'meta' / 'episodes'}")
    episodes = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    if episodes["episode_index"].duplicated().any():
        raise ValueError("Episode metadata contains duplicate episode_index values")
    return episodes.sort_values("episode_index").reset_index(drop=True)


def _scan_bounds(
    root: Path, episodes: pd.DataFrame, fps: int, tolerance: float
) -> dict[int, TrimBounds]:
    bounds: dict[int, TrimBounds] = {}
    expected_index = 0
    required = {"episode_index", "frame_index", "index", "timestamp", "action", "observation.state"}

    for path in _data_files(root):
        table = pq.read_table(path, columns=sorted(required))
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        episode_indices = table["episode_index"].to_numpy(zero_copy_only=False)
        starts = np.r_[0, np.flatnonzero(np.diff(episode_indices)) + 1]
        stops = np.r_[starts[1:], len(episode_indices)]
        actions = np.stack(table["action"].to_pylist())
        states = np.stack(table["observation.state"].to_pylist())
        frame_indices = table["frame_index"].to_numpy(zero_copy_only=False)
        global_indices = table["index"].to_numpy(zero_copy_only=False)
        timestamps = table["timestamp"].to_numpy(zero_copy_only=False)

        expected_global_indices = np.arange(expected_index, expected_index + len(table))
        if not np.array_equal(global_indices, expected_global_indices):
            raise ValueError(f"Non-contiguous global index in {path}")

        for start, stop in zip(starts, stops, strict=True):
            episode_index = int(episode_indices[start])
            if episode_index in bounds:
                raise ValueError(f"Episode {episode_index} appears in more than one data shard")
            length = stop - start
            if not np.array_equal(frame_indices[start:stop], np.arange(length)):
                raise ValueError(f"Frame indices are not contiguous for episode {episode_index}")
            expected_timestamp = np.arange(length, dtype=np.float32) / fps
            if not np.allclose(timestamps[start:stop], expected_timestamp, atol=1e-5):
                raise ValueError(f"Timestamps are invalid for episode {episode_index}")
            bounds[episode_index] = _trim_bounds(actions[start:stop], states[start:stop], tolerance)

        expected_index += len(table)

    metadata_episodes = set(episodes["episode_index"].astype(int))
    if set(bounds) != metadata_episodes:
        raise ValueError(
            f"Data/metadata episode mismatch: data={sorted(set(bounds) - metadata_episodes)}, "
            f"metadata={sorted(metadata_episodes - set(bounds))}"
        )
    return bounds


def _write_trim_report(output: Path, bounds: dict[int, TrimBounds]) -> None:
    rows = [
        {
            "episode_index": episode_index,
            "source_length": trim.source_length,
            "start_frame": trim.start,
            "end_frame": trim.end,
            "output_length": trim.end - trim.start,
            "trim_head": trim.start,
            "trim_tail": trim.source_length - trim.end,
            "tail_status": "release" if trim.has_release else "kept_no_release",
        }
        for episode_index, trim in sorted(bounds.items())
    ]
    frame_count = pd.DataFrame(rows)
    frame_count.to_csv(output / "meta" / "trimming.csv", index=False)


def _probe_video(path: Path, ffprobe: str) -> tuple[int, float]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    frame_count = int(stream["nb_read_frames"])
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return frame_count, float(numerator) / float(denominator)


def _find_binary(name: str) -> str:
    executable = shutil.which(name)
    if executable:
        return executable
    environment_binary = Path(sys.executable).with_name(name)
    if environment_binary.is_file():
        return str(environment_binary)
    raise RuntimeError(f"{name} must be available on PATH or next to the Python executable")


def _video_path(root: Path, video_template: str, video_key: str, row: pd.Series) -> Path:
    return root / video_template.format(
        video_key=video_key,
        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
        file_index=int(row[f"videos/{video_key}/file_index"]),
    )


def _encoder_args(info: dict) -> list[str]:
    extra_options = info.get("video.extra_options") or {}
    args = [
        "-c:v",
        "libx264",
        "-preset",
        str(info.get("video.preset", "veryfast")),
        "-crf",
        str(info.get("video.crf", 23)),
        "-pix_fmt",
        str(info.get("video.pix_fmt", "yuv420p")),
        "-g",
        str(info.get("video.g", 15)),
    ]
    for key, value in extra_options.items():
        args.extend([f"-{key}", str(value)])
    return args


def _encode_video(
    source: Path,
    output: Path,
    video_key: str,
    row: pd.Series,
    trim: TrimBounds,
    video_template: str,
    feature_info: dict,
    fps: int,
    ffmpeg: str,
    ffprobe: str,
) -> tuple[int, str, bool]:
    episode_index = int(row["episode_index"])
    source_path = _video_path(source, video_template, video_key, row)
    output_path = _video_path(output, video_template, video_key, row)
    expected_frames = trim.end - trim.start

    if output_path.is_file():
        try:
            frames, output_fps = _probe_video(output_path, ffprobe)
            if frames == expected_frames and np.isclose(output_fps, fps):
                return episode_index, video_key, True
        except (ValueError, KeyError, subprocess.CalledProcessError):
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.stem}.partial.mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-vf",
        f"trim=start_frame={trim.start}:end_frame={trim.end},setpts=PTS-STARTPTS",
        "-an",
        "-frames:v",
        str(expected_frames),
        "-r",
        str(fps),
        *_encoder_args(feature_info),
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        frames, output_fps = _probe_video(partial, ffprobe)
        if frames != expected_frames or not np.isclose(output_fps, fps):
            raise ValueError(
                f"{video_key} episode {episode_index}: encoded {frames} frames at {output_fps} fps, "
                f"expected {expected_frames} at {fps} fps"
            )
        partial.replace(output_path)
    except subprocess.CalledProcessError as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {source_path}:\n{error.stderr}") from error
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return episode_index, video_key, False


def _encode_videos(
    source: Path,
    output: Path,
    episodes: pd.DataFrame,
    bounds: dict[int, TrimBounds],
    info: dict,
    workers: int,
) -> None:
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    if not video_keys:
        raise ValueError("Dataset has no video features")

    jobs = []
    video_paths = set()
    for _, row in episodes.iterrows():
        for video_key in video_keys:
            path = _video_path(source, info["video_path"], video_key, row)
            if path in video_paths:
                raise ValueError(f"Multiple episodes reference the same video file: {path}")
            video_paths.add(path)
            jobs.append((video_key, row, bounds[int(row["episode_index"])]))

    ffmpeg = _find_binary("ffmpeg")
    ffprobe = _find_binary("ffprobe")

    reused = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _encode_video,
                source,
                output,
                video_key,
                row,
                trim,
                info["video_path"],
                info["features"][video_key]["info"],
                int(info["fps"]),
                ffmpeg,
                ffprobe,
            )
            for video_key, row, trim in jobs
        ]
        with tqdm(total=len(futures), desc="Trim videos", unit="video") as progress:
            for future in as_completed(futures):
                _, _, was_reused = future.result()
                reused += int(was_reused)
                progress.update(1)
    if reused:
        print(f"Reused {reused} valid output videos")


def _write_data(
    source: Path, output: Path, bounds: dict[int, TrimBounds], fps: int
) -> None:
    global_index = 0
    for source_path in _data_files(source):
        source_table = pq.read_table(source_path)
        frame_df = source_table.to_pandas()
        keep_rows = []
        for episode_index, group in frame_df.groupby("episode_index", sort=False):
            trim = bounds[int(episode_index)]
            keep_rows.extend(group.index[trim.start : trim.end].tolist())

        frame_df = frame_df.loc[keep_rows].copy()
        frame_df["frame_index"] = frame_df.groupby("episode_index", sort=False).cumcount()
        frame_df["timestamp"] = (
            frame_df["frame_index"].astype(np.float32) / fps
        ).astype(np.float32)
        frame_df["index"] = np.arange(global_index, global_index + len(frame_df), dtype=np.int64)
        global_index += len(frame_df)

        table = pa.Table.from_pandas(frame_df, preserve_index=False, schema=source_table.schema)
        output_path = output / source_path.relative_to(source)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_table_one_row_group_per_episode(table, output_path)

    expected = sum(trim.end - trim.start for trim in bounds.values())
    if global_index != expected:
        raise ValueError(f"Wrote {global_index} frames, expected {expected}")


def _write_episodes(
    source: Path,
    output: Path,
    episodes: pd.DataFrame,
    bounds: dict[int, TrimBounds],
    fps: int,
) -> None:
    metadata_paths = sorted((source / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if len(metadata_paths) != 1:
        raise ValueError(f"Expected one episode metadata shard, found {len(metadata_paths)}")
    source_table = pq.read_table(metadata_paths[0])
    episodes = episodes.copy()
    info = json.loads((source / "meta" / "info.json").read_text())
    video_keys = [
        key for key, feature in info["features"].items() if feature["dtype"] == "video"
    ]

    frame_offset = 0
    for index, row in episodes.iterrows():
        trim = bounds[int(row["episode_index"])]
        length = trim.end - trim.start
        episodes.at[index, "length"] = length
        episodes.at[index, "dataset_from_index"] = frame_offset
        episodes.at[index, "dataset_to_index"] = frame_offset + length
        for video_key in video_keys:
            episodes.at[index, f"videos/{video_key}/from_timestamp"] = 0.0
            episodes.at[index, f"videos/{video_key}/to_timestamp"] = length / fps
        frame_offset += length

    table = pa.Table.from_pandas(episodes, preserve_index=False, schema=source_table.schema)
    output_path = output / metadata_paths[0].relative_to(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    pq.write_table(table, output_path, compression="snappy")


def _verify_output(
    output: Path,
    episodes: pd.DataFrame,
    bounds: dict[int, TrimBounds],
    workers: int,
) -> None:
    info = json.loads((output / "meta" / "info.json").read_text())
    expected_frames = sum(trim.end - trim.start for trim in bounds.values())
    if int(info["total_frames"]) != expected_frames:
        raise ValueError(
            f"info.json total_frames={info['total_frames']}, expected {expected_frames}"
        )

    expected_episode = 0
    global_index = 0
    for path in _data_files(output):
        frame_df = pd.read_parquet(
            path,
            columns=["episode_index", "frame_index", "index", "timestamp"],
        )
        for episode_index, group in frame_df.groupby("episode_index", sort=False):
            if int(episode_index) != expected_episode:
                raise ValueError(
                    f"Expected episode {expected_episode}, found {episode_index} in {path}"
                )
            trim = bounds[int(episode_index)]
            length = trim.end - trim.start
            if len(group) != length:
                raise ValueError(
                    f"Episode {episode_index}: {len(group)} data rows, expected {length}"
                )
            if not np.array_equal(group["frame_index"].to_numpy(), np.arange(length)):
                raise ValueError(f"Episode {episode_index}: invalid frame_index")
            expected_timestamp = np.arange(length, dtype=np.float32) / int(info["fps"])
            if not np.allclose(group["timestamp"].to_numpy(), expected_timestamp, atol=1e-5):
                raise ValueError(f"Episode {episode_index}: invalid timestamp")
            expected_global = np.arange(global_index, global_index + length)
            if not np.array_equal(group["index"].to_numpy(), expected_global):
                raise ValueError(f"Episode {episode_index}: invalid global index")
            global_index += length
            expected_episode += 1

    if global_index != expected_frames or expected_episode != len(episodes):
        raise ValueError(f"Verified {global_index} frames/{expected_episode} episodes")

    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    ffprobe = _find_binary("ffprobe")

    def check_video(video_key: str, row: pd.Series) -> None:
        episode_index = int(row["episode_index"])
        trim = bounds[episode_index]
        path = _video_path(output, info["video_path"], video_key, row)
        frames, fps = _probe_video(path, ffprobe)
        if frames != trim.end - trim.start or not np.isclose(fps, info["fps"]):
            raise ValueError(
                f"{video_key} episode {episode_index}: {frames} frames at {fps} fps, "
                f"expected {trim.end - trim.start} at {info['fps']} fps"
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(check_video, video_key, row)
            for _, row in episodes.iterrows()
            for video_key in video_keys
        ]
        with tqdm(total=len(futures), desc="Verify videos", unit="video") as progress:
            for future in as_completed(futures):
                future.result()
                progress.update(1)


def main() -> None:
    args = _parse_args()
    source = args.source.resolve()
    output = (args.output or source.with_name(f"{source.name}_trimmed")).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if output == source:
        raise ValueError("--output must differ from --source")
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"Output already exists: {output}. Use --resume or choose another --output."
        )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    info = json.loads((source / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    episodes = _load_episodes(source)
    bounds = _scan_bounds(source, episodes, fps, args.start_tolerance)

    kept = sum(trim.end - trim.start for trim in bounds.values())
    removed = sum(trim.source_length - (trim.end - trim.start) for trim in bounds.values())
    no_release = sorted(ep for ep, trim in bounds.items() if not trim.has_release)
    print(
        f"Episode trim plan: {len(episodes)} episodes, {kept}/{kept + removed} frames kept "
        f"({kept / (kept + removed):.2%}), {removed} removed"
    )
    if no_release:
        print(f"No final gripper release (tail kept) for episodes: {no_release}")
    if args.dry_run:
        return

    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "meta", output / "meta", dirs_exist_ok=True)

    _encode_videos(source, output, episodes, bounds, info, args.workers)
    _write_data(source, output, bounds, fps)
    _write_episodes(source, output, episodes, bounds, fps)

    output_info = DatasetInfo.from_dict(info)
    output_info.total_frames = kept
    write_info(output_info, output)
    _write_trim_report(output, bounds)

    dataset = LeRobotDataset(repo_id=output.name, root=output)
    recompute_stats(dataset, skip_image_video=True)
    _verify_output(output, episodes, bounds, args.workers)
    print(f"Trimmed dataset written and verified: {output}")


if __name__ == "__main__":
    main()
