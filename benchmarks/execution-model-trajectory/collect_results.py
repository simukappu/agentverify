"""Run the execution-model x subject x scenario benchmark cells and emit results JSON + Markdown.

Invoked by ``run_all.sh``. Can also be run directly:

    python benchmarks/execution-model-trajectory/collect_results.py \
        --output-json benchmarks/execution-model-trajectory/results/results-YYYY-MM-DDTHHMMSS.json \
        --output-md   benchmarks/execution-model-trajectory/results/results-YYYY-MM-DDTHHMMSS.md

Two scenarios are supported (see ``DESIGN.md`` for what they mean):

- ``dev``: Scenario 1, "first run with no cassette / no fixture". All cells drive the agent against a real LLM.
- ``ci``: Scenario 2, "PR-time repeat run". Only the agentverify cells skip the LLM (cassette replay); B and C run the agent live, since DeepEval and AgentCore Evaluations have no documented CI-mode mechanism that bypasses the agent runtime.

Use ``--scenario {dev,ci,all}`` (default ``all``).

LOC, API call count, dollar cost, CI secrets, and cold-start overhead are recorded as static values per (subject, model, scenario). See README.md "Reproducing the measurement" for the full procedure.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

# Cells are described by (subject, model). The two scenarios share the same test file; the test functions are differentiated by the ``_dev`` / ``_ci`` suffix.
CELLS: list[dict[str, Any]] = [
    {
        "subject": "strands-weather-forecaster",
        "execution_model": "A",
        "label": "agentverify (inline / pytest / SDK patching)",
        "test_file": "strands-weather-forecaster/agentverify_test.py",
        "needs_aws": True,  # Scenario 1 calls Bedrock; Scenario 2 cassette replay does not. Captured per-scenario in needs_llm.
        "needs_llm_per_scenario": {"dev": True, "ci": False},
    },
    {
        "subject": "strands-weather-forecaster",
        "execution_model": "B",
        "label": "DeepEval (`@observe` + ToolCorrectnessMetric)",
        "test_file": "strands-weather-forecaster/deepeval_test.py",
        "needs_aws": True,
        "needs_llm_per_scenario": {"dev": True, "ci": True},
    },
    {
        "subject": "strands-weather-forecaster",
        "execution_model": "C",
        "label": "AgentCore Evaluations Custom code-based evaluator (Lambda)",
        "test_file": "strands-weather-forecaster/agentcore_test.py",
        "needs_aws": True,
        "needs_llm_per_scenario": {"dev": True, "ci": True},
    },
    {
        "subject": "langgraph-multi-agent-supervisor",
        "execution_model": "A",
        "label": "agentverify (inline / pytest / SDK patching)",
        "test_file": "langgraph-multi-agent-supervisor/agentverify_test.py",
        "needs_aws": False,
        "needs_llm_per_scenario": {"dev": True, "ci": False},
    },
    {
        "subject": "langgraph-multi-agent-supervisor",
        "execution_model": "B",
        "label": "DeepEval (`@observe` + ToolCorrectnessMetric)",
        "test_file": "langgraph-multi-agent-supervisor/deepeval_test.py",
        "needs_aws": False,
        "needs_llm_per_scenario": {"dev": True, "ci": True},
    },
    {
        "subject": "langgraph-multi-agent-supervisor",
        "execution_model": "C",
        "label": "AgentCore Evaluations Custom code-based evaluator (Lambda)",
        "test_file": "langgraph-multi-agent-supervisor/agentcore_test.py",
        "needs_aws": True,
        "needs_llm_per_scenario": {"dev": True, "ci": True},
    },
]

# Static measurements per (subject, model, scenario). LOC is auto-counted at runtime from the test files; the rest are filled in here. When the implementation changes, update this table to match.
#
# CI secrets:
#   - A dev (Strands): AWS credentials (Bedrock invoke). 1 secret.
#   - A ci (Strands): none. 0 secrets.
#   - A dev (LangGraph): OPENAI_API_KEY. 1 secret.
#   - A ci (LangGraph): none. 0 secrets.
#   - B dev/ci (Strands): AWS (Bedrock invoke) + OPENAI_API_KEY (DeepEval metric construction). 2 secrets.
#   - B dev/ci (LangGraph): OPENAI_API_KEY (covers both supervisor and DeepEval). 1 secret.
#   - C dev/ci (Strands): AWS (Bedrock + AgentCore Evaluations `Evaluate` + Lambda invoke). 1 secret.
#   - C dev/ci (LangGraph): AWS + OPENAI_API_KEY. 2 secrets.
#
# Dollar cost per run (LLM tokens dominate; Lambda + AgentCore Evaluations `Evaluate` < $0.000001 each):
#   - Strands LLM run (Bedrock Sonnet 4.6): 12277 input + 667 output tokens (cassette aggregate)
#       = 12277 * 3.00/1e6 + 667 * 15.00/1e6 = $0.0468
#   - LangGraph LLM run (gpt-5.4-mini): 5226 input + 320 output tokens (cassette aggregate)
#       = 5226 * 0.75/1e6 + 320 * 4.50/1e6 = $0.0054
#   - A ci: cassette replay, no LLM, $0.
#   - C cells: same LLM cost as the same-subject B cell, plus negligible Lambda + AgentCore Evaluations `Evaluate` (about $0.000001 / call), rounded to the same 4-decimal display.
#   - Pricing snapshot date: 2026-05-17 (Bedrock Anthropic Claude Sonnet 4.6 $3 in / $15 out per 1M; OpenAI gpt-5.4-mini $0.75 in / $4.50 out per 1M).
STATIC_METRICS: dict[str, dict[str, Any]] = {
    # Strands
    "strands-weather-forecaster:A:dev": {"api_calls_per_run": 3, "dollar_cost_per_run_usd": 0.0468, "ci_secrets_required": 1},
    "strands-weather-forecaster:A:ci": {"api_calls_per_run": 0, "dollar_cost_per_run_usd": 0.0000, "ci_secrets_required": 0},
    "strands-weather-forecaster:B:dev": {"api_calls_per_run": 3, "dollar_cost_per_run_usd": 0.0468, "ci_secrets_required": 2},
    "strands-weather-forecaster:B:ci": {"api_calls_per_run": 3, "dollar_cost_per_run_usd": 0.0468, "ci_secrets_required": 2},
    "strands-weather-forecaster:C:dev": {"api_calls_per_run": 4, "dollar_cost_per_run_usd": 0.0468, "ci_secrets_required": 1},
    "strands-weather-forecaster:C:ci": {"api_calls_per_run": 4, "dollar_cost_per_run_usd": 0.0468, "ci_secrets_required": 1},
    # LangGraph (cassette has 10 LLM calls with gpt-5.4-mini)
    "langgraph-multi-agent-supervisor:A:dev": {"api_calls_per_run": 10, "dollar_cost_per_run_usd": 0.0054, "ci_secrets_required": 1},
    "langgraph-multi-agent-supervisor:A:ci": {"api_calls_per_run": 0, "dollar_cost_per_run_usd": 0.0000, "ci_secrets_required": 0},
    "langgraph-multi-agent-supervisor:B:dev": {"api_calls_per_run": 10, "dollar_cost_per_run_usd": 0.0054, "ci_secrets_required": 1},
    "langgraph-multi-agent-supervisor:B:ci": {"api_calls_per_run": 10, "dollar_cost_per_run_usd": 0.0054, "ci_secrets_required": 1},
    "langgraph-multi-agent-supervisor:C:dev": {"api_calls_per_run": 11, "dollar_cost_per_run_usd": 0.0054, "ci_secrets_required": 2},
    "langgraph-multi-agent-supervisor:C:ci": {"api_calls_per_run": 11, "dollar_cost_per_run_usd": 0.0054, "ci_secrets_required": 2},
}


# ---------------------------------------------------------------------------
# Wall-time measurement
# ---------------------------------------------------------------------------


def _run_pytest(test_rel_path: str, scenario: str, repo_root: Path, env_extra: dict[str, str]) -> tuple[bool, float | None, str]:
    """Run a single pytest invocation for one (cell, scenario), return (passed, duration_seconds, stdout).

    Wall time is measured with ``time.perf_counter()`` around the subprocess so the recorded value preserves sub-millisecond precision (pytest's ``--durations`` output rounds to two decimals, which collapses the third digit). The measurement therefore includes pytest process startup + plugin init in addition to the test body; this overhead is uniform across all 12 cells, so cell-to-cell comparisons remain valid.
    """
    venv = Path(env_extra.get("AGENTVERIFY_BENCH_VENV") or (Path.home() / ".venvs" / "agentverify-bench"))
    pytest_bin = venv / "bin" / "pytest"
    bench_pyproject = ROOT / "pyproject.toml"
    cmd = [
        str(pytest_bin),
        "-c",
        str(bench_pyproject),
        f"benchmarks/execution-model-trajectory/{test_rel_path}",
        "-k",
        scenario,
        "-v",
        "--no-header",
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env={**dict(os.environ), **env_extra},
        check=False,
    )
    elapsed = time.perf_counter() - start
    passed = proc.returncode == 0 and "PASSED" in proc.stdout
    duration = elapsed if passed else None
    return passed, duration, proc.stdout + proc.stderr


def _measure_cell(cell: dict[str, Any], scenario: str, repo_root: Path, runs: int = 5) -> dict[str, Any]:
    """Measure wall time for a single (cell, scenario) over ``runs`` invocations."""
    label = f"{cell['subject']}:{cell['execution_model']}:{scenario}"
    print(f"[bench] {label}: {cell['label']}")
    durations: list[float] = []
    failures: list[str] = []
    for i in range(runs):
        passed, duration, stdout = _run_pytest(cell["test_file"], scenario, repo_root, env_extra={})
        if not passed:
            failures.append(_redact_paths(stdout[-2000:]))
        elif duration is not None:
            durations.append(duration)
        print(
            f"  run {i + 1}/{runs}: "
            + (f"passed in {duration:.3f}s" if passed and duration is not None else "FAILED")
        )

    return {
        "wall_time_runs_seconds": durations,
        "wall_time_trimmed_mean_seconds": _trimmed_mean(durations),
        "wall_time_median_seconds": statistics.median(durations) if durations else None,
        "wall_time_min_seconds": min(durations) if durations else None,
        "wall_time_max_seconds": max(durations) if durations else None,
        "passed_count": len(durations),
        "failed_count": runs - len(durations),
        "failure_excerpts": failures,
    }


def _trimmed_mean(values: list[float]) -> float | None:
    """Return the mean after dropping the single highest and lowest entries.

    Returns ``None`` for fewer than 3 values, since trimming both ends would leave nothing to average. With 3 values the trimmed mean is the median; the formal definition still applies.
    """
    if len(values) < 3:
        return None
    sorted_values = sorted(values)
    middle = sorted_values[1:-1]
    return statistics.fmean(middle)


# ---------------------------------------------------------------------------
# LOC counting
# ---------------------------------------------------------------------------

_BENCH_BLOCK_RE = re.compile(
    r"^[ \t]*#\s*---\s*benchmark assertion\s*\(LOC counted:[^)]*\)[^\n]*\n(?P<body>.*?)^[ \t]*#\s*---\s*end benchmark assertion[^\n]*",
    re.MULTILINE | re.DOTALL,
)


def _count_assertion_loc(test_path: Path) -> int | None:
    """Count assertion-essential LOC in a test file.

    The implementation looks for paired ``# --- benchmark assertion ...`` and ``# --- end benchmark assertion ...`` markers and counts the non-blank, non-comment lines between them. The Strands and LangGraph cells share a single ``_assert_trajectory`` helper between the ``_dev`` and ``_ci`` test wrappers, so the helper is counted once per cell and applies to both scenarios; the wrapper test bodies (which only differ in how they obtain the trajectory) are intentionally not counted.

    Returns ``None`` if no marker block is found (e.g. the file structure changed).
    """
    if not test_path.exists():
        return None
    text = test_path.read_text(encoding="utf-8")
    blocks = list(_BENCH_BLOCK_RE.finditer(text))
    if not blocks:
        return None
    total = 0
    for block in blocks:
        for raw in block.group("body").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            total += 1
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_PATH_RE = re.compile(r"/(?:Users|home)/[^/\s]+/")


def _redact_paths(text: str) -> str:
    """Replace absolute user-home paths with a redaction marker."""
    return _USER_PATH_RE.sub("/<redacted>/", text)


def _environment_snapshot() -> dict[str, Any]:
    """Capture environment metadata recorded alongside the results."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    versions: dict[str, str | None] = {}
    try:
        import importlib.metadata as md  # type: ignore

        for name in (
            "agentverify",
            "deepeval",
            "boto3",
            "aws-cdk-lib",
            "aws-cdk.aws-bedrock-agentcore-alpha",
            "strands-agents",
            "langgraph",
            "langchain-openai",
        ):
            try:
                versions[name] = md.version(name)
            except md.PackageNotFoundError:
                versions[name] = None
    except Exception as exc:  # pragma: no cover
        versions["_error"] = str(exc)
    info["package_versions"] = versions

    info["aws_cdk_cli"] = _aws_cdk_cli_version()
    info["aws_cli"] = _aws_cli_version()
    return info


def _aws_cdk_cli_version() -> str | None:
    npx = shutil.which("npx")
    if not npx:
        return None
    try:
        proc = subprocess.run(
            [npx, "--yes", "aws-cdk@2", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 else None
    except Exception:
        return None


def _aws_cli_version() -> str | None:
    aws = shutil.which("aws")
    if not aws:
        return None
    try:
        proc = subprocess.run(
            [aws, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return (proc.stdout or proc.stderr).strip().splitlines()[0]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def _format_markdown(data: dict[str, Any]) -> str:
    env = data["environment"]
    lines: list[str] = []
    lines.append("# Execution Model Comparison Benchmark — Results")
    lines.append("")
    if data.get("preliminary"):
        lines.append("> **Preliminary**: not yet treated as the canonical reference for blog / CHANGELOG citation. Drop the `preliminary-` prefix and this banner once a run is promoted to canonical.")
        lines.append("")
    lines.append(f"**Run timestamp (UTC):** {data['timestamp']}")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Platform: `{env['platform']}` ({env['machine']})")
    lines.append(f"- Python: `{env['python_version']}`")
    lines.append(f"- CPU count: {env['cpu_count']}")
    if env.get("aws_cli"):
        lines.append(f"- AWS CLI: `{env['aws_cli']}`")
    if env.get("aws_cdk_cli"):
        lines.append(f"- AWS CDK CLI: `{env['aws_cdk_cli']}`")
    lines.append("")
    lines.append("Package versions:")
    for name, version in env["package_versions"].items():
        lines.append(f"- `{name}` = `{version}`")
    lines.append("")

    # Per-scenario tables.
    for scenario in data["scenarios_run"]:
        lines.append(f"## Scenario {1 if scenario == 'dev' else 2} ({scenario}) — wall time (seconds, trimmed mean of middle 3 of 5)")
        lines.append("")
        lines.append("| Subject | Execution model | Trimmed mean | Median | Min | Max | All runs | Pass/Fail |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cell in data["cells"]:
            if cell["scenario"] != scenario:
                continue
            runs = cell["wall_time_runs_seconds"]
            trimmed = cell.get("wall_time_trimmed_mean_seconds")
            median = cell["wall_time_median_seconds"]
            min_v = cell.get("wall_time_min_seconds")
            max_v = cell.get("wall_time_max_seconds")
            runs_str = ", ".join(f"{r:.3f}" for r in runs)
            passfail = f"{cell['passed_count']}/{cell['failed_count']}"
            trimmed_str = f"{trimmed:.3f}" if trimmed is not None else "—"
            median_str = f"{median:.3f}" if median is not None else "—"
            min_str = f"{min_v:.3f}" if min_v is not None else "—"
            max_str = f"{max_v:.3f}" if max_v is not None else "—"
            lines.append(
                f"| {cell['subject']} | {cell['execution_model']} ({cell['label']}) | "
                f"{trimmed_str} | {median_str} | {min_str} | {max_str} | {runs_str} | {passfail} |"
            )
        lines.append("")

    lines.append("## Static metrics (per scenario)")
    lines.append("")
    lines.append("| Subject | Execution model | Scenario | LOC | API calls / run | $ / run | CI secrets |")
    lines.append("|---|---|---|---|---|---|---|")
    for cell in data["cells"]:
        s = cell["static"]
        lines.append(
            f"| {cell['subject']} | {cell['execution_model']} | {cell['scenario']} | "
            f"{cell.get('loc') or '—'} | {s['api_calls_per_run']} | "
            f"${s['dollar_cost_per_run_usd']:.4f} | {s['ci_secrets_required']} |"
        )
    lines.append("")

    lines.append("## Cold-start observations")
    lines.append("")
    obs = data.get("cold_start_observations") or {}
    measurements = obs.get("measurements") or {}
    if measurements:
        lines.append("| Subject | Execution model | Scenario | Cold-start (s) |")
        lines.append("|---|---|---|---|")
        for key, value in measurements.items():
            parts = key.split(":")
            subject = parts[0] if len(parts) > 0 else key
            model = parts[1] if len(parts) > 1 else ""
            scen = parts[2] if len(parts) > 2 else "ci"
            lines.append(f"| {subject} | {model} | {scen} | {value:.3f} |")
    else:
        lines.append("Not recorded for this run. Pass `--cold-start-seconds SUBJECT:MODEL:SCENARIO=SECONDS` to `collect_results.py` (or to `run_all.sh`) after capturing the cold-start wall time per the README's step 7.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Two scenarios are recorded: ``dev`` (Scenario 1, first run with no cassette) and ``ci`` (Scenario 2, PR-time repeat). See `DESIGN.md` for what each measures.")
    lines.append("- LOC counts only assertion-essential lines (no imports, no boilerplate). The benchmark uses paired `# --- benchmark assertion (LOC counted: ...) ---` markers in each test file. See DESIGN.md for the counting rule.")
    lines.append("- LOC for execution model C is the sum of the test-side assertion block (the OTLP payload construction in `agentcore_test.py`) and the Lambda-side assertion block (`agentcore_evaluator_cdk/lambda_src/lambda_function.py`). Both move together when the assertion changes, so they are counted together. CDK / IAM / deploy-script wiring is one-time setup and is not included; see DESIGN.md \"Setup LOC\" for that breakdown.")
    lines.append("- Wall time is the trimmed mean of the middle 3 runs out of 5 (single highest and lowest dropped). The full 5-run series is in the `All runs` column.")
    lines.append("- `$ / run` is dominated by the LLM-token cost. Per-LLM-call cost is computed from the cassette token aggregates and current public list pricing (Bedrock Anthropic Claude Sonnet 4.6: $3 input / $15 output per 1M tokens; OpenAI gpt-5.4-mini: $0.75 input / $4.50 output per 1M tokens; pricing snapshot 2026-05-17).")
    lines.append("- B and C cells have the same `$ / run` as long as they invoke the same LLM the same number of times, even though C also issues an `Evaluate` API call against AgentCore Evaluations (which then internally invokes the assertion Lambda). AWS Lambda invocation and CloudWatch Logs ingestion together come to about $0.0000005 per Evaluate at the configured Lambda size, which rounds to zero at the table's 4-decimal display. AgentCore Evaluations does not publish a documented per-call price for the data-plane `Evaluate` call as of the pricing snapshot date; the table treats it as zero pending official guidance, which is a known understatement.")
    lines.append("- agentverify's CI scenario is exactly $0 because cassette replay does not call the LLM.")
    lines.append("- Cold-start wall time is **not** in the warm wall-time table; it lives in the \"Cold-start observations\" section above when recorded.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument(
        "--timestamp",
        default=None,
        help="ISO 8601 UTC timestamp recorded in the results files (e.g., 2026-05-17T21:30:45Z). Defaults to the current UTC time when not provided.",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--scenario",
        choices=("dev", "ci", "all"),
        default="all",
        help="Which scenario(s) to run.",
    )
    parser.add_argument(
        "--skip-aws",
        action="store_true",
        help="Skip cells that hit AWS (execution model C, plus any cell whose Scenario 1 invokes Bedrock).",
    )
    parser.add_argument(
        "--preliminary",
        action="store_true",
        help="Mark the run as preliminary (development laptop). Emits a banner at the top of the Markdown output indicating that the numbers are not the canonical figures.",
    )
    parser.add_argument(
        "--cold-start-seconds",
        action="append",
        default=[],
        metavar="SUBJECT:MODEL:SCENARIO=SECONDS",
        help=(
            "Record a cold-start wall-time observation. Example: "
            "--cold-start-seconds strands-weather-forecaster:C:ci=1.40"
        ),
    )
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    cold_start = _parse_cold_start_args(args.cold_start_seconds)

    repo_root = ROOT.parents[1]

    scenarios = ("dev", "ci") if args.scenario == "all" else (args.scenario,)

    cells: list[dict[str, Any]] = []
    for cell in CELLS:
        for scenario in scenarios:
            if args.skip_aws and cell["needs_aws"]:
                print(f"[bench] skip {cell['subject']}:{cell['execution_model']}:{scenario} (--skip-aws)")
                continue
            measurement = _measure_cell(cell, scenario, repo_root, runs=args.runs)
            static_key = f"{cell['subject']}:{cell['execution_model']}:{scenario}"
            test_path = ROOT / cell["test_file"]
            loc = _count_assertion_loc(test_path)
            # For execution model C, add the Lambda-side assertion LOC: each PR-time assertion change touches both the test-side payload construction and the Lambda-side walker, so the per-assertion cost is the sum.
            if cell["execution_model"] == "C" and loc is not None:
                lambda_path = (
                    ROOT
                    / cell["subject"]
                    / "agentcore_evaluator_cdk"
                    / "lambda_src"
                    / "lambda_function.py"
                )
                lambda_loc = _count_assertion_loc(lambda_path)
                if lambda_loc is not None:
                    loc = loc + lambda_loc
            cells.append(
                {
                    **{k: v for k, v in cell.items() if k != "needs_llm_per_scenario"},
                    "scenario": scenario,
                    "needs_llm": cell["needs_llm_per_scenario"][scenario],
                    **measurement,
                    "loc": loc,
                    "static": STATIC_METRICS[static_key],
                }
            )

    cold_start_observations: dict[str, Any] = {
        "_doc": "Cold-start wall times are the wall time of the first invocation against a freshly-initialised Lambda execution environment. See README.md step 7 for how to force a cold init and capture the value.",
    }
    if cold_start:
        cold_start_observations["measurements"] = cold_start

    data = {
        "timestamp": timestamp,
        "preliminary": args.preliminary,
        "scenarios_run": list(scenarios),
        "environment": _environment_snapshot(),
        "cells": cells,
        "cold_start_observations": cold_start_observations,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(data, f, indent=2, default=str)
    with open(args.output_md, "w") as f:
        f.write(_format_markdown(data))

    return 0


def _parse_cold_start_args(values: list[str]) -> dict[str, float]:
    """Parse --cold-start-seconds entries of the form SUBJECT:MODEL:SCENARIO=SECONDS."""
    out: dict[str, float] = {}
    for entry in values:
        if "=" not in entry:
            raise SystemExit(
                f"--cold-start-seconds expects SUBJECT:MODEL:SCENARIO=SECONDS, got {entry!r}"
            )
        key, _, value = entry.partition("=")
        try:
            out[key.strip()] = float(value.strip())
        except ValueError:
            raise SystemExit(
                f"--cold-start-seconds value must be a number, got {value!r} (in {entry!r})"
            )
    return out


if __name__ == "__main__":
    sys.exit(main())
