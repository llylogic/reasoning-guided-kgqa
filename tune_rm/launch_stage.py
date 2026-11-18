"""Train one stage and merge it before the next stage, using paths.json."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import resolved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", type=int, choices=(1, 2, 3))
    p.add_argument("dataset", choices=("webqsp", "cwq"))
    p.add_argument("--paths", default=str(ROOT / "paths.json"))
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--accelerate-config", default=str(ROOT / "default_config.yaml"))
    p.add_argument("--single-device", action="store_true")
    p.add_argument(
        "--train-file",
        help="optional prepared stage file, e.g. an archived JSON dataset",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="resume if the stage adapter was saved",
    )
    a = p.parse_args()
    paths = resolved(a.paths)
    work = Path(paths["output_dir"]) / a.dataset
    base = paths["backbone"] if a.stage == 1 else work / f"merged{a.stage - 1}"
    adapter = work / f"adapter{a.stage}"
    merged = work / f"merged{a.stage}"
    if merged.exists():
        raise FileExistsError(merged)
    if not Path(base).is_dir():
        raise FileNotFoundError(base)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if not a.merge_only:
        launch = (
            [sys.executable]
            if a.single_device
            else [
                "accelerate",
                "launch",
                "--config_file",
                str(Path(a.accelerate_config).resolve()),
            ]
        )
        subprocess.run(
            launch
            + [
                str(ROOT / "run_clm_finetune.py"),
                "--stage",
                str(a.stage),
                "--dataset",
                a.dataset,
                "--paths",
                str(Path(a.paths).resolve()),
                "--config",
                str(Path(a.config).resolve()),
            ]
            + (
                ["--train-file", str(Path(a.train_file).resolve())]
                if a.train_file
                else []
            ),
            env=env,
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tune_rm/merge.py"),
            str(base),
            str(adapter),
            str(merged),
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
