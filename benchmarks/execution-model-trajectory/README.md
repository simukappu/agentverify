# Assertion Execution Model Comparison Benchmark

This benchmark expresses the **same execution trajectory assertion** in three different assertion execution models and records measurable differences. The goal is to make the comparison concrete with numbers rather than narrative.

The three models compared are inline (agentverify), separate-runner (DeepEval), and managed remote (Amazon Bedrock AgentCore Evaluations). Throughout the docs and code, *execution model* is the short form of *assertion execution model*; both refer to the same thing.

For the methodology, the assertion design, the LOC counting rule, and the explicit non-goals, see [`DESIGN.md`](./DESIGN.md). This README is the operating manual.

## Two scenarios per cell

Every (execution model × subject) cell is measured under two scenarios:

- **Scenario 1 - dev / first run**: the agent has never been exercised against this assertion before. The test must drive the agent end-to-end against a real LLM. All three execution models pay the same agent runtime + LLM cost; what differs is the framework overhead.
- **Scenario 2 - CI / repeat run**: the same assertion runs every PR. Each execution model uses its own documented CI pattern, and the benchmark measures what each model can avoid re-paying.

The Scenario 2 execution differs sharply between models: agentverify replays the canonical cassette and skips the LLM entirely, while DeepEval and AgentCore Evaluations have no comparable mechanism documented for CI use, so under Scenario 2 they run the agent end-to-end against the LLM exactly as they do under Scenario 1. The benchmark deliberately does not invent a trajectory-fixture path for B or C; doing so would lend agentverify's cassette capability to the other two execution models in a shape no real DeepEval / AgentCore Evaluations user is expected to write. See [`DESIGN.md`](./DESIGN.md#two-scenarios-why-the-benchmark-needs-both) for the full reasoning and the concrete data sources for each cell.

## Subjects under test

Two existing examples are reused as agents under test, matching the framework set that Amazon Bedrock AgentCore Evaluations supports at the SDK level:

- [`examples/strands-weather-forecaster/`](../../examples/strands-weather-forecaster/) — single-agent, two-step ReAct against the US National Weather Service (NWS) API.
- [`examples/langgraph-multi-agent-supervisor/`](../../examples/langgraph-multi-agent-supervisor/) — multi-agent supervisor with `add` chaining.

Each subject has a thin bench-local agent factory at `<subject>/_<framework>_agent_factory.py` (`_strands_agent_factory.py` and `_langgraph_agent_factory.py`) that pins the model id and centralises agent construction. All three execution models import from it, so they exercise the same agent. The factory module names are unique across the two subjects so that running both in a single pytest invocation does not collide on a shared module name.

## The assertion (per subject)

All three execution models must enforce the same logical check for a given subject. Both subject-level assertions deliberately exercise tool ordering, argument matching, and cross-step data flow, so that the AgentCore Evaluations execution model goes through its Custom code-based evaluator (Lambda) rather than the cheaper built-in trajectory matcher path. The [`DESIGN.md`](./DESIGN.md#why-the-assertion-has-to-go-beyond-tool-name-ordering) explains why the built-in matcher does not satisfy the comparison.

**Strands Weather Forecaster (single-agent, two-step ReAct):**

> Step 0 calls `http_request` with `method=GET` and a URL matching `/points/`. Step 1 calls `http_request` with `method=GET` and a URL matching `/forecast`. The URL used in step 1 contains data returned by step 0's response body (cross-step data flow).

**LangGraph Multi-Agent Supervisor (multi-agent handoff with `add` chain):**

> Supervisor hands off to `research_expert` first. The research step issues at least one `web_search` whose query mentions FAANG / headcount / employees. Control returns to the supervisor, which hands off to `math_expert`. The second `add` step's `a` argument equals the first `add` step's result (running-total flow across steps).

## Execution models

| ID | Execution model | Implementation |
|---|---|---|
| A | agentverify (inline, pytest, SDK patching) | `<subject>/agentverify_test.py`. Scenario 1 drives the agent under `cassette(mode="record", cassette_dir=tmp)` and asserts on the resulting `ExecutionResult`. Scenario 2 replays the canonical cassette under `examples/<subject>/tests/cassettes/`. |
| B | DeepEval deterministic mode (`@observe` + `ToolCorrectnessMetric`) | `<subject>/deepeval_test.py` decorates the agent driver with `@observe`, runs it against the LLM, and asserts via `ToolCorrectnessMetric` plus a custom deterministic check. Same code path under both scenarios; B has no documented CI mechanism that bypasses the LLM. |
| C | AgentCore Evaluations Custom code-based evaluator (Lambda on-demand) | `<subject>/agentcore_evaluator_cdk/` packages a Lambda that asserts on a recorded session. The `<subject>/agentcore_test.py` driver runs the agent in-process, observes its tool calls via the framework's callback / message walk, packs them into an OTLP span attribute, and calls the AgentCore Evaluations data-plane `Evaluate` API. Same code path under both scenarios. |

## Repository layout

```
benchmarks/execution-model-trajectory/
  README.md                                  # this file
  DESIGN.md                                  # methodology, assertion rationale, non-goals
  pyproject.toml                             # local pytest config (importmode=importlib)
  run_all.sh                                 # drive both subjects across both scenarios, emit unified results
  collect_results.py                         # measurement loop invoked by run_all.sh
  strands-weather-forecaster/                # primary subject (Strands)
    _strands_agent_factory.py                # bench-local agent factory (model id pinned)
    conftest.py                              # registers the agentverify cassette fixture
    agentverify_test.py                      # execution model A (Scenario 1 + Scenario 2)
    deepeval_test.py                         # execution model B (Scenario 1 + Scenario 2)
    agentcore_test.py                        # execution model C driver (Scenario 1 + Scenario 2)
    agentcore_evaluator_cdk/                 # execution model C: CDK app
      app.py
      stack.py
      cdk.json
      requirements.txt
      deploy.sh
      destroy.sh
      lambda_src/
        lambda_function.py
  langgraph-multi-agent-supervisor/          # secondary subject (LangGraph)
    _langgraph_agent_factory.py
    conftest.py
    agentverify_test.py
    deepeval_test.py
    agentcore_test.py
    agentcore_evaluator_cdk/
      ...
  results/
    results-YYYY-MM-DDTHHMMSS.json
    results-YYYY-MM-DDTHHMMSS.md
```

This directory is not part of the published PyPI artifact; the `[tool.setuptools.packages.find] include = ["agentverify*"]` rule in `pyproject.toml` keeps it out.

## Test naming convention

Each `*_test.py` file ships two test functions with the suffixes `_dev` and `_ci`:

- `test_<execution_model>_<subject>_dev` runs the Scenario 1 path.
- `test_<execution_model>_<subject>_ci` runs the Scenario 2 path.

For B and C the two functions resolve to the same code path (B and C have no LLM-skipping CI mechanism), but they remain separate test functions so pytest selectors and the driver can address them independently. For A the two functions diverge: `_dev` records a fresh cassette to a `tmp_path`, `_ci` replays the canonical example cassette.

## How to run

### Setup

The benchmark uses a dedicated venv (Python 3.14) so the dependency set for execution models B and C does not pollute the project's working venv:

```bash
# from repo root
uv venv ~/.venvs/agentverify-bench --python 3.14 --seed
~/.venvs/agentverify-bench/bin/pip install -e ".[bedrock,dev]"
~/.venvs/agentverify-bench/bin/pip install -e "examples/strands-weather-forecaster"
~/.venvs/agentverify-bench/bin/pip install -e "examples/langgraph-multi-agent-supervisor"
~/.venvs/agentverify-bench/bin/pip install deepeval strands-agents-tools  # execution model B
~/.venvs/agentverify-bench/bin/pip install aws-cdk-lib aws-cdk.aws-bedrock-agentcore-alpha boto3  # execution model C
```

The benchmark directory ships its own `pyproject.toml` with `importmode=importlib`, which prevents pytest module-name collisions between the two subjects (both define `agentverify_test.py` / `deepeval_test.py` / `agentcore_test.py`). When invoking pytest manually below, pass `-c benchmarks/execution-model-trajectory/pyproject.toml` so this config applies; the `run_all.sh` driver does this automatically.

For execution model C, the per-subject `agentcore_evaluator_cdk/deploy.sh` script wraps `npx aws-cdk@2 deploy` and writes a stack outputs JSON file that `agentcore_test.py` reads. AWS resources are tagged `project=agentverify-execution-model-bench` for cleanup; tear down with the matching `destroy.sh`.

### LLM credentials

Scenario 1 (and the B / C cells under Scenario 2) drive a real LLM. The credentials each cell needs are:

| Cell | Scenario 1 secrets | Scenario 2 secrets |
|---|---|---|
| Strands A | AWS credentials with Bedrock invoke permission on `us.anthropic.claude-sonnet-4-6` | none (cassette replay) |
| Strands B | AWS credentials with Bedrock invoke permission, `OPENAI_API_KEY` (DeepEval `@observe` requires it at metric construction time even in deterministic mode) | same as Scenario 1 |
| Strands C | AWS credentials with Bedrock invoke + AgentCore Evaluations `Evaluate` + Lambda invoke permission | same as Scenario 1 |
| LangGraph A | `OPENAI_API_KEY` for `gpt-5.4-mini` | none (cassette replay) |
| LangGraph B | `OPENAI_API_KEY` (used by both the agent and DeepEval) | same as Scenario 1 |
| LangGraph C | `OPENAI_API_KEY`, AWS credentials with AgentCore Evaluations `Evaluate` + Lambda invoke permission | same as Scenario 1 |

Total LLM-token cost across all 12 cells × 5 runs is approximately $1.30 at current public pricing. The dominant term is 5 Strands cells × 5 runs at about $0.047/run (about $1.17 total); the LangGraph side adds about $0.14 (5 cells × 5 runs at about $0.0054/run). Lambda and AgentCore Evaluations `Evaluate` per-call charges are below a tenth of a cent each and round to zero at the table's 4-decimal display.

### Drive both subjects across both scenarios

```bash
cd benchmarks/execution-model-trajectory
./run_all.sh
```

The driver writes timestamped results into `results/`:

- `results-YYYY-MM-DDTHHMMSS.json` (machine-readable, all 12 cells)
- `results-YYYY-MM-DDTHHMMSS.md` (table per scenario + observation bullets)

`./run_all.sh --scenario dev` and `./run_all.sh --scenario ci` run only one scenario; the default `--scenario all` runs both back-to-back.

### Run one cell directly

| Cell | Command |
|---|---|
| Strands A dev | `~/.venvs/agentverify-bench/bin/pytest -c benchmarks/execution-model-trajectory/pyproject.toml benchmarks/execution-model-trajectory/strands-weather-forecaster/agentverify_test.py -k dev` |
| Strands A ci | `... -k ci` (replace the `-k` filter; same file) |
| Strands B dev / ci | `... benchmarks/execution-model-trajectory/strands-weather-forecaster/deepeval_test.py -k dev` / `-k ci` |
| Strands C dev / ci | first `cd benchmarks/execution-model-trajectory/strands-weather-forecaster/agentcore_evaluator_cdk && ./deploy.sh`, then `... benchmarks/execution-model-trajectory/strands-weather-forecaster/agentcore_test.py -k dev` / `-k ci`, finally `./destroy.sh` |
| LangGraph * | analogous, swapping `strands-weather-forecaster` for `langgraph-multi-agent-supervisor` |

## Reproducing the measurement

To run the bench during development, pass `--preliminary` to `run_all.sh`. The driver writes the results files with a `preliminary-` filename prefix and a banner at the top of the Markdown output, which marks the run as not yet promoted to the canonical reference for blog / CHANGELOG citation. Drop the prefix and the banner once a run is treated as canonical (or run without `--preliminary` from the start).

The full procedure is below. Steps 1-3 are one-time host setup; steps 4-5 are one-time AWS setup; steps 6-9 are the measurement loop.

### Prerequisites on the target machine (steps 1-3)

1. **Install host tooling**:
   - [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for venv management.
   - [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) for credential setup and CFN inspection.
   - Node.js LTS (Node 20, 22, or 24 at the time of writing). Newer non-LTS versions print a JSII warning but still run; older or unsupported versions can fail outright. The CDK CLI itself is fetched on demand by the deploy / destroy scripts via `npx --yes aws-cdk@2`; no global `cdk` install is needed.

2. **Set up the bench venv** as in [Setup](#setup) above. Verify with `~/.venvs/agentverify-bench/bin/python --version` printing `Python 3.14.x`.

3. **Configure credentials** for the two providers and for AWS:
   - `aws sts get-caller-identity` should return an account where the principal can invoke Bedrock (Sonnet 4) and create / update / delete Lambda, IAM, CloudFormation, CloudWatch, AgentCore Evaluations. Choose a region where AgentCore Evaluations is available; the benchmark defaults to `us-east-1`. Set `AWS_REGION` if you want to override.
   - `OPENAI_API_KEY` set to a real key. The LangGraph subject and DeepEval's `ToolCorrectnessMetric` both check for it at construction time.

### One-time AWS setup (steps 4-5)

4. **Bootstrap CDK in the target account x region** (one time per account x region; can be skipped if you have already bootstrapped CDK there before, e.g. for any other CDK project):

   ```bash
   ( cd benchmarks/execution-model-trajectory/strands-weather-forecaster/agentcore_evaluator_cdk \
     && PATH="$HOME/.venvs/agentverify-bench/bin:$PATH" \
        npx --yes aws-cdk@2 bootstrap --app "python app.py" )
   ```

   Either CDK directory's `app.py` works for bootstrap; the result is shared across both stacks. The bootstrap creates a `CDKToolkit` CloudFormation stack with a staging S3 bucket and a few IAM roles. Persistent footprint is on the order of a few cents per month (S3 storage for CDK assets).

5. **Deploy both AgentCore Evaluations stacks** (independent stacks, tagged for cleanup). Run from the repo root:

   ```bash
   ( cd benchmarks/execution-model-trajectory/strands-weather-forecaster/agentcore_evaluator_cdk && ./deploy.sh )
   ( cd benchmarks/execution-model-trajectory/langgraph-multi-agent-supervisor/agentcore_evaluator_cdk && ./deploy.sh )
   ```

   Each `deploy.sh` writes an `outputs.json` next to itself; the corresponding `agentcore_test.py` reads it.

### Measurement loop (steps 6-9)

6. **Quiesce the machine**: close other heavy applications. Use a stable, low-jitter network connection. Disable scheduled background tasks (Spotlight indexing on macOS, Time Machine, system updates) for the duration of the run.

7. **Capture cold-start wall time** for each AgentCore Evaluations stack. AWS Lambda guarantees a new (cold) execution environment after any `UpdateFunctionConfiguration` API call, so bump each function's description to a fresh value, then time the next invocation:

   ```bash
   # Force a cold init by updating the Lambda configuration. The description bump is benign and idempotent — it only affects the function's metadata, not its behaviour or its IAM role.
   FN_W=agentverify-execution-model-bench-weather
   FN_L=agentverify-execution-model-bench-langgraph
   STAMP=$(date -u +%s)
   aws lambda update-function-configuration --function-name "$FN_W" --description "force-cold-$STAMP" --region us-east-1
   aws lambda update-function-configuration --function-name "$FN_L" --description "force-cold-$STAMP" --region us-east-1
   # Wait until both functions return to Active state before the next invocation.
   aws lambda wait function-updated --function-name "$FN_W" --region us-east-1
   aws lambda wait function-updated --function-name "$FN_L" --region us-east-1

   # Time the first (cold) invocation per stack. Use the ci scenario so the wall time excludes agent runtime.
   AWS_REGION=us-east-1 ~/.venvs/agentverify-bench/bin/pytest \
     -c benchmarks/execution-model-trajectory/pyproject.toml \
     benchmarks/execution-model-trajectory/strands-weather-forecaster/agentcore_test.py -k ci \
     --durations=0 -v
   AWS_REGION=us-east-1 ~/.venvs/agentverify-bench/bin/pytest \
     -c benchmarks/execution-model-trajectory/pyproject.toml \
     benchmarks/execution-model-trajectory/langgraph-multi-agent-supervisor/agentcore_test.py -k ci \
     --durations=0 -v
   ```

   Note the test-call duration printed by `--durations=0` for each subject. These are the cold-start values; pass them to the driver in step 8 via `--cold-start-seconds`.

8. **Run the driver** for both scenarios (5 invocations per cell × 2 scenarios = 10 wall-time measurements per cell). Pass the cold-start values you captured in step 7 so they are recorded alongside the warm numbers:

   ```bash
   AWS_REGION=us-east-1 ./benchmarks/execution-model-trajectory/run_all.sh \
     --cold-start-seconds strands-weather-forecaster:C=<seconds> \
     --cold-start-seconds langgraph-multi-agent-supervisor:C=<seconds>
   ```

   The driver writes `results/results-YYYY-MM-DDTHHMMSS.{json,md}`. Confirm the JSON's `environment.platform`, `environment.machine`, and `environment.package_versions` match the target machine.

9. **Tear down** when measurements are complete:

   ```bash
   ( cd benchmarks/execution-model-trajectory/strands-weather-forecaster/agentcore_evaluator_cdk && ./destroy.sh )
   ( cd benchmarks/execution-model-trajectory/langgraph-multi-agent-supervisor/agentcore_evaluator_cdk && ./destroy.sh )
   ```

   Optionally remove the CDK bootstrap stack (only if no other CDK project in this account x region needs it):

   ```bash
   aws cloudformation delete-stack --stack-name CDKToolkit --region us-east-1
   ```

### What the numbers depend on

- **Network**: AgentCore Evaluations C cells round-trip to the AWS region. Network jitter is the dominant variance for those cells. Same-region client-side execution (e.g., on an EC2 instance in `us-east-1`) would cut the network component but is not the realistic CI scenario, so the benchmark deliberately runs the client off-AWS.
- **CPU**: Scenario 2 / A cells are dominated by Python startup, pytest plugin init, cassette parse, and the assertion. Scenario 1 cells across all execution models are dominated by LLM round-trip plus agent runtime; CPU generation matters less for these.
- **Region**: keep the AWS region constant across runs.
- **Tooling versions**: the driver records `agentverify`, `deepeval`, `aws-cdk.aws-bedrock-agentcore-alpha`, `strands-agents`, `langgraph`, and `langchain-openai` versions in `results-YYYY-MM-DDTHHMMSS.json`. Pin them in your bench venv before the run so future measurements are apples-to-apples.

## Metrics collected

Per (subject × execution model × scenario) cell:

| Metric | Unit | Source |
|---|---|---|
| Implementation LOC (test + wiring) | lines | non-blank, non-comment lines inside the paired `# --- benchmark assertion ...` markers in each test file. The Strands and LangGraph cells share a single `_assert_trajectory` helper between `_dev` and `_ci`, so LOC is reported once per (subject, model) and applies to both scenarios. For C, the test-side assertion block and the Lambda-side assertion block in `agentcore_evaluator_cdk/lambda_src/lambda_function.py` move together when the assertion changes and are reported as their sum. CDK / IAM / deploy-script wiring is one-time setup; see [`DESIGN.md`](./DESIGN.md#setup-loc-footnote-not-primary) for that breakdown |
| Test execution wall time (trimmed mean of middle 3 of 5) | s | 5 invocations are timed; the single highest and lowest are dropped, the remaining 3 are averaged. `pytest --durations=0` produces the per-run numbers |
| API calls per test run | count | manual enumeration of LLM provider calls + AWS API calls. For C, this counts the boto3 `Evaluate` API call once; the Lambda invocation that AgentCore Evaluations makes inside that call is an internal AWS hop and is not counted as a client-issued API call |
| Dollar cost per test run | USD | LLM-token cost computed from cassette token aggregates × current public pricing (Bedrock Sonnet 4.6: $3 in / $15 out per 1M; OpenAI gpt-5.4-mini: $0.75 in / $4.50 out per 1M; snapshot 2026-05-17). B and C cells have identical `$ / run` whenever they invoke the same LLM the same number of times, because the LLM-token cost dominates and the additional AgentCore Evaluations + Lambda overhead in C is below the 4-decimal display threshold (Lambda invocation alone is about $0.0000005 at the configured size; AgentCore Evaluations `Evaluate` does not publish a documented per-call price as of the snapshot date, and the table treats it as zero pending official guidance — a known understatement) |
| CI secrets required | count | distinct credentials the CI job must hold (drops to zero for A under Scenario 2) |
| Cold start overhead | s | for C, includes Lambda init duration; for A/B it is the time to first assertion |

The LOC counting rule is documented in [`DESIGN.md`](./DESIGN.md#loc-counting-rule) so it is consistent across all 12 cells. Each results file pins the exact rule and tooling versions it was measured with.

## Non-goals

Summarised in [`DESIGN.md`](./DESIGN.md#non-goals); the short version:

- No LLM-as-a-Judge benchmarking (out of scope: this benchmark only compares deterministic, code-based checks).
- **No trajectory-fixture path for B or C.** Inventing one would compare a benchmark scaffold rather than the products' documented CI patterns.
- No load / scaling tests.
- No accuracy comparison on a correct run (all three execution models are deterministic by design).
- Not wired into CI on every commit.
