#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <pretrained_model_dir> [output_dir]" >&2
    echo "Default output: <pretrained_model_dir>_migrated" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

if [[ ! -d "$1" ]]; then
    echo "Checkpoint directory not found: $1" >&2
    exit 1
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
input_dir=$(realpath "$1")
output_dir=$(realpath -m "${2:-${input_dir%/}_migrated}")

if [[ "$input_dir" == "$output_dir" ]]; then
    echo "Output directory must differ from the input checkpoint." >&2
    exit 1
fi

for required in config.json model.safetensors; do
    if [[ ! -s "$input_dir/$required" ]]; then
        echo "Missing checkpoint file: $input_dir/$required" >&2
        exit 1
    fi
done

python_bin=${PYTHON:-python}
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python not found. Activate the lerobot environment or set PYTHON=/path/to/python." >&2
    exit 1
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" "$repo_root/src/lerobot/processor/migrate_policy_normalization.py" \
    --pretrained-path "$input_dir" \
    --output-dir "$output_dir" \

for required in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [[ ! -s "$output_dir/$required" ]]; then
        echo "Migration did not produce: $output_dir/$required" >&2
        exit 1
    fi
done

"$python_bin" - "$output_dir" <<'PY'
from pathlib import Path
import sys

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors

path = Path(sys.argv[1])
config = PreTrainedConfig.from_pretrained(path)
config.device = "cpu"
policy = get_policy_class(config.type).from_pretrained(path, config=config)
make_pre_post_processors(config, pretrained_path=str(path))
print(f"Load check passed: {type(policy).__name__} from {path}")
PY
