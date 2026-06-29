import argparse
import itertools
import json
import re
import signal
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run train.py over all hyperparameter combinations in a YAML-like sweep config."
    )
    parser.add_argument("--config", type=str, default="sweep_config.yaml")
    parser.add_argument("--train-script", type=str, default="train.py")
    parser.add_argument("--state-file", type=str, default="sweep_progress.json")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def parse_simple_yaml(path: Path) -> OrderedDict:
    """
    Parse a small YAML subset used by this project:
      - top-level `key: value`
      - top-level `key:` followed by list items `- value`
    """
    data: OrderedDict = OrderedDict()
    current_list_key = None

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            is_indented = len(line) != len(line.lstrip())
            if is_indented:
                if current_list_key is None:
                    raise ValueError(
                        f"{path}:{lineno} unexpected indentation without a list key."
                    )
                item = stripped
                if not item.startswith("-"):
                    raise ValueError(f"{path}:{lineno} expected list item like '- value'.")
                item_value = item[1:].strip()
                data[current_list_key].append(parse_scalar(item_value))
                continue

            current_list_key = None
            if ":" not in stripped:
                raise ValueError(f"{path}:{lineno} expected 'key: value' format.")

            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(f"{path}:{lineno} empty key is not allowed.")

            if value == "":
                data[key] = []
                current_list_key = key
            else:
                data[key] = parse_scalar(value)

    return data


def read_train_args(train_script: Path) -> set:
    content = train_script.read_text(encoding="utf-8")
    # Accept both single and double quotes in add_argument("--name")
    names = re.findall(r'add_argument\(\s*[\'"]--([A-Za-z0-9_]+)[\'"]', content)
    return set(names)


def normalize_grid(config: OrderedDict) -> Tuple[List[str], List[List]]:
    keys = list(config.keys())
    value_lists = []
    for key in keys:
        value = config[key]
        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError(f"Config key '{key}' has an empty list.")
            value_lists.append(value)
        else:
            value_lists.append([value])
    return keys, value_lists


def get_varying_keys(config: OrderedDict) -> List[str]:
    varying_keys = []
    for key, value in config.items():
        if isinstance(value, list) and len(value) > 1:
            varying_keys.append(key)
    return varying_keys


def make_run_name(params: Dict, varying_keys: List[str]) -> str:
    if not varying_keys:
        return "default"

    parts = []
    for key in varying_keys:
        value = params.get(key)
        parts.append(f"{key}-{value}")
    return "-".join(parts)


def to_run_id(params: Dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def load_state(state_path: Path) -> Dict:
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}, "failed": {}, "in_progress": None}


def save_state(state_path: Path, state: Dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_command(
    python_exe: str, train_script: Path, params: Dict, run_name: str
) -> List[str]:
    cmd = [python_exe, str(train_script)]
    for key, value in params.items():
        if value is None:
            continue
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--run_name", run_name])
    return cmd


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    train_script = Path(args.train_script).resolve()
    state_path = Path(args.state_file).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")

    config = parse_simple_yaml(config_path)
    valid_args = read_train_args(train_script)
    unknown = [k for k in config.keys() if k not in valid_args]
    if unknown:
        raise ValueError(
            f"Unknown config keys for {train_script.name}: {unknown}. "
            f"Allowed args: {sorted(valid_args)}"
        )

    keys, value_lists = normalize_grid(config)
    varying_keys = get_varying_keys(config)
    combos = [dict(zip(keys, values)) for values in itertools.product(*value_lists)]

    state = load_state(state_path)
    completed = state.get("completed", {})
    failed = state.get("failed", {})
    in_progress = state.get("in_progress")

    total = len(combos)
    print(f"Loaded {total} sweep combinations from {config_path}.")
    if in_progress:
        print(f"Last interrupted in-progress run found: {in_progress.get('run_id')}")

    interrupted = False

    def handle_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handle_sigint)

    for idx, params in enumerate(combos, start=1):
        run_id = to_run_id(params)
        if not args.force_rerun and run_id in completed:
            print(f"[{idx}/{total}] SKIP completed: {params}")
            continue

        run_name = make_run_name(params, varying_keys)
        cmd = build_command(args.python, train_script, params, run_name)
        pretty_cmd = " ".join(cmd)
        print(f"[{idx}/{total}] RUN: {params}")
        print(f"Run name: {run_name}")
        print(f"Command: {pretty_cmd}")

        if args.dry_run:
            print("Dry-run mode: command not executed.")
            continue

        state["in_progress"] = {
            "run_id": run_id,
            "params": params,
            "command": cmd,
            "started_at": now_iso(),
        }
        save_state(state_path, state)

        try:
            result = subprocess.run(cmd, cwd=train_script.parent)
        except KeyboardInterrupt:
            interrupted = True
            result = None

        if interrupted:
            print("Interrupted by user. Progress saved; completed runs will be skipped next time.")
            save_state(state_path, state)
            return 130

        if result is not None and result.returncode == 0:
            completed[run_id] = {
                "params": params,
                "command": cmd,
                "completed_at": now_iso(),
            }
            state["completed"] = completed
            state["in_progress"] = None
            save_state(state_path, state)
            print(f"[{idx}/{total}] DONE")
            continue

        failed[run_id] = {
            "params": params,
            "command": cmd,
            "failed_at": now_iso(),
            "return_code": None if result is None else result.returncode,
        }
        state["failed"] = failed
        state["in_progress"] = None
        save_state(state_path, state)
        print(f"[{idx}/{total}] FAILED with return code {result.returncode if result else 'n/a'}")

        if not args.continue_on_error:
            return 1

    print("Sweep completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
