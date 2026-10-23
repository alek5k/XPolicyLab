#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
    echo "Usage: $0 bench_name ckpt_name env_cfg_type joint /path/to/official-v30-export" >&2
    exit 1
fi
[[ "$4" == joint ]] || { echo 'Only joint actions are supported' >&2; exit 1; }
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir=$(realpath "$5")
[[ -f "$source_dir/meta/info.json" ]] || { echo 'Missing LeRobot meta/info.json' >&2; exit 1; }
python -c 'import json, sys; m=json.load(open(sys.argv[1])); assert m["codebase_version"] == "v3.0", "Expected official LeRobot v3.0 export"; required={"action", "observation.state", "observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"}; assert required <= m["features"].keys(), "Missing official SmolVLA camera/state/action keys"' "$source_dir/meta/info.json"
mkdir -p "$POLICY_DIR/data"
ln -s "$source_dir" "$POLICY_DIR/data/$1-$2-$3-$4"
echo '[LTE_SmolVLA] Linked official export; normalization is fitted by training processors.'
